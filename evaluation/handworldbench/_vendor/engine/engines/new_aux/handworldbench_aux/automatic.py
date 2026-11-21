from __future__ import annotations
import gc, os, sys, types
from .core import ROOT, DATA, model_path, stage_model, MetricIssue, result, load_vendor
from .video import sample
_VISUAL_PLAUSIBILITY_MODELS = {}

class NativeVideoProcessor:
    """Trace-only adapter: never alters upstream preprocessing arguments or pixels."""

    def __init__(self, processor):
        self.processor = processor
        self.last_input = {}
        d = processor.video_processor.to_dict()
        self.settings = {k: d.get(k) for k in ['fps', 'size', 'max_frames', 'min_frames', 'patch_size', 'temporal_patch_size', 'merge_size', 'cap_pixels_per_frame']}

    def apply_chat_template(self, messages, **kwargs):
        batch = self.processor.apply_chat_template(messages, **kwargs)
        self.last_input = {k: list(v.shape) for k, v in batch.items() if hasattr(v, 'shape')}
        if 'video_grid_thw' in batch:
            self.last_input['video_grid_thw_values'] = batch['video_grid_thw'].tolist()
        return batch
