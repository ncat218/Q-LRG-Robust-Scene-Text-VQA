"""Extract a single rectangular ROI using the public 7B selector checkpoints."""

import argparse
import hashlib
import sys
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, __version__, set_seed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qlrg.common import object_hash, read_json, write_json
from qlrg.experiment import paired_images
from scripts.sdrpn_select import RegionSelected, capture_region


class Selector:
    def __init__(self, vendor, checkpoint, processor):
        if __version__ != "4.51.3":
            raise RuntimeError(f"Use Transformers 4.51.3 for Vision-RL², found {__version__}")
        sys.path.insert(0, str(vendor))
        from qwen_src.roi import heatmap
        from qwen_src.qwen2_5_vl import modeling_qwen2_5_vl_batch as upstream

        config = read_json(checkpoint / "config.json")
        if (config["twig_K"], config["twig_T"], config["roi_source"]) != (18, 3, "qk"):
            raise ValueError("Expected the public Qwen2.5-VL-7B K18/T3 selector")

        def intercept(maps, images, _processor, _encoder, grid, threshold, **_kwargs):
            if len(maps) != 1 or maps[0] is None:
                raise ValueError("Expected a single ROI map from the 7B selector")
            capture_region(maps[0].unsqueeze(0), images, grid, threshold, heatmap)

        upstream.get_batched_sub_images_v2 = intercept
        self.processor = AutoProcessor.from_pretrained(
            processor, local_files_only=True, use_fast=True, min_pixels=3136, max_pixels=288 * 28 * 28)
        self.processor.tokenizer.padding_side = "left"
        model, loading = upstream.Qwen2_5_VLForConditionalGeneration.from_pretrained(
            checkpoint, local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
            use_safetensors=True, output_loading_info=True,
        )
        if any(loading.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")):
            raise RuntimeError(f"7B selector checkpoint failed to load: {loading}")
        self.model = model.to("cuda").eval().requires_grad_(False)
        backbone = self.model.model
        if not backbone.enable_twig or not config.get("roi_multi_head"):
            raise RuntimeError("Checkpoint did not enable the multi-head RoI predictor")
        self.model.roi_enable2stage = backbone.roi_enable2stage = True
        backbone.roi_conf_thresh = 0.15

    @torch.inference_mode()
    def locate(self, image, text):
        model = self.model
        backbone = model.model
        model.rope_deltas = None
        backbone.last_pos_id = None
        backbone.generation_attention_mask = None
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        batch = self.processor(text=[prompt], images=[image], return_tensors="pt").to("cuda")
        try:
            model.generate(**batch, src_images=[image], processor=self.processor,
                           max_new_tokens=1, do_sample=False, use_cache=True)
        except RegionSelected as region:
            return {"box": region.box, "foreground_tokens": region.tokens}
        raise RuntimeError("7B selector did not reach the ROI prefill hook")


def run(args):
    request = read_json(args.request)
    set_seed(request["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    selector = Selector(args.vendor, args.checkpoint, args.processor)
    for index, record in enumerate(request["images"], 1):
        shard = args.output / f"{object_hash(record['image_id'])[:24]}.json"
        if not shard.is_file():
            _, lr4 = paired_images(Path(request["textvqa"]), record)
            results = {}
            for question in record["questions"]:
                torch.cuda.synchronize()
                start = time.perf_counter()
                region = selector.locate(lr4, request["prompt"].format(question=question["query"]))
                torch.cuda.synchronize()
                results[str(question["question_id"])] = {**region, "selector_seconds": time.perf_counter() - start}
            write_json(shard, {"image_id": record["image_id"], "questions": results,
                               "lr4_sha256": hashlib.sha256(lr4.tobytes()).hexdigest()})
        if index % 50 == 0 or index == len(request["images"]):
            print(f"{args.output.name} proposals: {index}/{len(request['images'])} images", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("request", "output", "vendor", "checkpoint", "processor"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    run(parser.parse_args())
