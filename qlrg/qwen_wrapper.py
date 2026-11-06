"""Pinned Qwen2.5-VL inference and D0 identity checks."""

import inspect
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import transformers

from .common import sha256
from .feature_injection import inject_features
from .grid_alignment import image_grid


class Qwen:
    def __init__(self, config, hf_hub):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.config = config
        source = inspect.getfile(Qwen2_5_VLForConditionalGeneration)
        if transformers.__version__ != config["transformers"] or sha256(source) != config["source_sha256"]:
            raise RuntimeError("Transformers version or Qwen source differs from the audited implementation")
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("D0/S0 requires a CUDA allocation with BF16")
        self.device = torch.device("cuda:0")
        self.dtype = getattr(torch, config["dtype"])
        self.snapshot = Path(hf_hub) / ("models--" + config["id"].replace("/", "--")) / "snapshots" / config["revision"]
        if not (self.snapshot / "config.json").is_file():
            raise FileNotFoundError(f"Pinned model snapshot unavailable: {self.snapshot}")
        self.processor = AutoProcessor.from_pretrained(
            self.snapshot, local_files_only=True, use_fast=True,
            min_pixels=config["min_pixels"], max_pixels=config["max_pixels"],
        )
        if not self.processor.image_processor.__class__.__name__.endswith("Fast"):
            raise RuntimeError("Expected the pinned fast image processor")
        self.processor.tokenizer.padding_side = "left"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.snapshot, local_files_only=True, dtype=self.dtype,
            attn_implementation=config["attention"],
        ).to(self.device).eval().requires_grad_(False)

    def prepare(self, image, question):
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": self.config["prompt"].format(question=question)},
        ]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        batch = self.processor(text=[prompt], images=[image], return_tensors="pt")
        batch = {name: tensor.to(self.device) for name, tensor in batch.items()}
        h, w = image_grid(batch["image_grid_thw"][0].tolist(), self.model.model.visual.spatial_merge_size)
        if int((batch["input_ids"] == self.model.config.image_token_id).sum()) != h * w:
            raise RuntimeError("Image placeholder count differs from the merged grid")
        return batch

    @torch.inference_mode()
    def encode(self, batch):
        (feature,) = self.model.model.get_image_features(batch["pixel_values"], batch["image_grid_thw"])
        return feature

    def replacement(self, batch, feature):
        if feature is None:
            return nullcontext()
        return inject_features(self.model.model, batch["image_grid_thw"], (feature,))

    @torch.inference_mode()
    def generate(self, batch, feature=None):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with self.replacement(batch, feature):
            output = self.model.generate(
                **batch, do_sample=False, num_beams=1, use_cache=True,
                max_new_tokens=self.config["max_new_tokens"],
                repetition_penalty=self.config["repetition_penalty"],
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        tokens = output[0, batch["input_ids"].shape[1]:].tolist()
        answer = self.processor.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return {"answer": answer.strip(), "token_ids": tokens, "generation_seconds": elapsed}

    @torch.inference_mode()
    def compare(self, native_batch, injected_batch, feature):
        """Require exact prefill logits and generated token IDs for parity controls."""
        expected = self.model(**native_batch, use_cache=False, logits_to_keep=1).logits
        with self.replacement(injected_batch, feature):
            actual = self.model(**injected_batch, use_cache=False, logits_to_keep=1).logits
        max_diff = float((expected.float() - actual.float()).abs().max())
        native = self.generate(native_batch)
        injected = self.generate(injected_batch, feature)
        if not torch.equal(expected, actual) or native["token_ids"] != injected["token_ids"]:
            raise RuntimeError(f"D0 injection parity failed (prefill max abs difference {max_diff})")
        return {"max_abs_logit_difference": max_diff, "generated_tokens_equal": True}


def assert_aligned(clean, degraded):
    for key in ("image_grid_thw", "input_ids", "attention_mask"):
        if not torch.equal(clean[key], degraded[key]):
            raise RuntimeError(f"Clean and LR4 conditions are misaligned: {key}")
