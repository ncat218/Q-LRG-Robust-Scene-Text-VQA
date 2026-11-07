"""Run the SD-RPN prefill in its own Transformers environment and record crop boxes."""

import argparse
import hashlib
import sys
import time
from pathlib import Path

import torch
from torchvision.transforms.functional import gaussian_blur
from transformers import AutoProcessor, __version__, set_seed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qlrg.common import object_hash, read_json, write_json
from qlrg.experiment import paired_images


class RegionSelected(Exception):
    def __init__(self, box, tokens):
        self.box = box
        self.tokens = tokens


def capture_region(roi, images, grid, threshold, helpers):
    if len(images) != 1 or roi.ndim != 3 or tuple(roi.shape[:1]) != (1,) or not torch.isfinite(roi).all():
        raise ValueError("Expected one finite SD-RPN ROI map")
    h, w = (int(v) // 2 for v in grid[0, 1:])
    if (h, w) != tuple(roi.shape[1:]) or h * w > 288:
        raise ValueError("SD-RPN ROI map does not match the expected visual grid")
    kernel = [3 if w > 1 else 1, 3 if h > 1 else 1]
    mask = gaussian_blur(roi.sigmoid().unsqueeze(1), kernel_size=kernel, sigma=1.0)[0, 0] > threshold
    count = int(mask.sum().item())
    if not count:
        raise RegionSelected(None, 0)
    region = helpers.get_foreground_bbox_torch(mask.float().unsqueeze(0), threshold=0.5)
    box = [int(x) for x in helpers.get_bbox4src(region, images[0], feat_size=(h, w), img_size=(h * 28, w * 28))[0]]
    width, height = images[0].size
    if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
        raise ValueError(f"Invalid SD-RPN crop for {(width, height)}: {box}")
    raise RegionSelected(box, count)


class Selector:
    def __init__(self, vendor, checkpoint, processor_source):
        if __version__ != "4.51.3":
            raise RuntimeError(f"Use the SD-RPN environment with Transformers 4.51.3, found {__version__}")
        sys.path.insert(0, str(vendor))
        from qwen_src import mm_utils
        from qwen_src.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
        from qwen_src.qwen2_5_vl import modeling_qwen2_5_vl as upstream

        def intercept(roi, images, _processor, _encoder, grid, threshold=0.15, *_args, **_kwargs):
            capture_region(roi, images, grid, threshold, mm_utils)

        self.processor = AutoProcessor.from_pretrained(
            processor_source, local_files_only=True, use_fast=True, min_pixels=3136, max_pixels=288 * 28 * 28,
        )
        self.processor.tokenizer.padding_side = "left"
        model, loading = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            checkpoint, local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
            use_safetensors=True, output_loading_info=True,
        )
        if any(loading.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")):
            raise RuntimeError(f"SD-RPN checkpoint failed to load: {loading}")
        self.model = model.to("cuda").eval().requires_grad_(False)
        if not self.model.model.enable_twig or not self.model.model.roi_multi_head:
            raise RuntimeError("Checkpoint did not enable the SD-RPN query-conditioned ROI head")
        self.model.model.roi_conf_thresh = 0.15
        self.model.model.is_debug = False
        upstream.get_batched_sub_images = intercept

    @torch.inference_mode()
    def locate(self, image, text):
        model = self.model
        backbone = model.model
        model.rope_deltas = None
        backbone.last_pos_id = None
        backbone.generation_attention_mask = None
        if hasattr(backbone, "rope_deltas"):
            delattr(backbone, "rope_deltas")
        backbone.enable_twig = backbone.roi_enable2stage = backbone.two_stage_vanilla = True
        for index, layer in enumerate(backbone.layers):
            layer.self_attn.layer_idx = index
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        batch = self.processor(text=[prompt], images=[image], return_tensors="pt").to("cuda")
        try:
            model.generate(**batch, src_images=[image], processor=self.processor,
                           max_new_tokens=1, do_sample=False, use_cache=True)
        except RegionSelected as region:
            return {"box": region.box, "foreground_tokens": region.tokens}
        raise RuntimeError("SD-RPN did not reach its ROI prefill hook")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("request", "output", "vendor", "checkpoint", "processor"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    request = read_json(args.request)
    selector = None
    for index, record in enumerate(request["images"], start=1):
        shard = args.output / f"{object_hash(record['image_id'])[:24]}.json"
        if shard.is_file():
            continue
        if selector is None:
            set_seed(request["seed"])
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
            selector = Selector(args.vendor, args.checkpoint, args.processor)
        _, lr4 = paired_images(Path(request["textvqa"]), record)
        start = time.perf_counter()
        results = {}
        for question in record["questions"]:
            qid = str(question["question_id"])
            results[qid] = {
                route: selector.locate(lr4, request["prompt"].format(question=question[route]))
                for route in ("query", "shuffled")
            }
        write_json(shard, {"image_id": record["image_id"], "lr4_sha256": hashlib.sha256(lr4.tobytes()).hexdigest(),
                           "selector_seconds": time.perf_counter() - start, "questions": results})
        if index % 50 == 0 or index == len(request["images"]):
            print(f"SD-RPN proposals: {index}/{len(request['images'])} images", flush=True)


if __name__ == "__main__":
    main()
