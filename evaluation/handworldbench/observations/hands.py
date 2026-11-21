"""Text-conditioned hand detection and CUDA keypoint inference."""
import inspect
import time
import numpy as np

PROMPTS = {"human": "hand.", "handson": "robotic hand. mechanical hand.", "real_robot": "robotic hand. mechanical hand."}

def box_iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    intersection = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    area = np.maximum(0, a[2:] - a[:2]).prod() + np.maximum(0, b[2:] - b[:2]).prod() - intersection
    return float(intersection / max(area, 1e-12))

def suppress(boxes, scores, threshold=0.5):
    keep = []
    for j in sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i)):
        if not any(box_iou(boxes[j], boxes[k]) > threshold for k in keep):
            keep.append(j)
    return keep

class HandDetector:
    def __init__(self, weights, device="cuda:0", batch=8):
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        self.processor = AutoProcessor.from_pretrained(weights, local_files_only=True)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(weights, local_files_only=True).to(device).eval()
        self.device, self.batch = device, batch
        self.threshold_arg = "box_threshold" if "box_threshold" in inspect.signature(self.processor.post_process_grounded_object_detection).parameters else "threshold"

    def predict(self, images, category):
        import torch
        start = time.monotonic()
        result, offset, batches, reductions = [], 0, 0, 0
        while offset < len(images):
            block = images[offset:offset+self.batch]
            try:
                inputs = self.processor(images=block, text=[PROMPTS[category]]*len(block), padding=True, return_tensors="pt").to(self.device)
                with torch.inference_mode():
                    outputs = self.model(**inputs)
                decoded = self.processor.post_process_grounded_object_detection(outputs, inputs.input_ids, text_threshold=0.25, target_sizes=[im.size[::-1] for im in block], **{self.threshold_arg: 0.25})
            except torch.cuda.OutOfMemoryError:
                if self.batch == 1:
                    raise
                self.batch = max(1, self.batch//2)
                reductions += 1
                torch.cuda.empty_cache()
                continue
            if len(decoded) != len(block):
                raise ValueError("Detection batch output count mismatch")
            for im, det in zip(block, decoded):
                boxes, scores = det["boxes"].detach().cpu().numpy(), det["scores"].detach().cpu().numpy()
                boxes[:, [0,2]] = boxes[:, [0,2]].clip(0, im.width)
                boxes[:, [1,3]] = boxes[:, [1,3]].clip(0, im.height)
                keep = suppress(boxes, scores)
                result.append([dict(box=boxes[j].tolist(), detection_confidence=float(scores[j])) for j in keep if np.isfinite(boxes[j]).all() and np.all(boxes[j,2:] > boxes[j,:2])])
            offset += len(block)
            batches += 1
        return result, dict(seconds=time.monotonic()-start, batch=self.batch, batches=batches, oom_reductions=reductions, prompt=PROMPTS[category])

class KeypointEstimator:
    def __init__(self, weights, batch=32, require_cuda=True):
        # Initialise CUDA through PyTorch before ONNX Runtime: on some drivers, creating the ONNX Runtime CUDA session
        # first in a fresh worker process crashes in the PTX JIT of the driver (segmentation fault). Results are unchanged.
        import torch
        if torch.cuda.is_available():
            torch.zeros(1, device="cuda")
        import onnxruntime as ort
        from pathlib import Path
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(Path(weights).read_bytes(), sess_options=options, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.providers = self.session.get_providers()
        if require_cuda and (not self.providers or self.providers[0] != "CUDAExecutionProvider"):
            raise RuntimeError(f"RTMPose CUDA unavailable; actual providers: {self.providers}")
        self.input = self.session.get_inputs()[0]
        self.batch = batch if not isinstance(self.input.shape[0], int) else self.input.shape[0]
        self.dynamic_batch = not isinstance(self.input.shape[0], int)

    def predict(self, images, detections):
        import cv2
        from .pose_geometry import bbox_xyxy2cs, top_down_affine, postprocess
        start = time.monotonic()
        outputs = [[dict(d) for d in frame] for frame in detections]
        jobs = []
        for i, (image, frame) in enumerate(zip(images, detections)):
            pixels = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            for j, det in enumerate(frame):
                center, scale = bbox_xyxy2cs(np.asarray(det["box"],float),padding=1.25)
                crop, scale = top_down_affine((256,256),scale,center,pixels)
                crop = (crop-np.array([123.675,116.28,103.53]))/np.array([58.395,57.12,57.375])
                jobs.append((i,j,crop.transpose(2,0,1).astype("float32"),center,scale))
        offset = batches = 0
        while offset < len(jobs):
            block = jobs[offset:offset+self.batch]
            array = np.stack([job[2] for job in block])
            actual_count = len(block)
            if not self.dynamic_batch and actual_count < self.batch:
                array = np.concatenate([array, np.repeat(array[-1:],self.batch-actual_count,axis=0)])
            try:
                values = self.session.run(None,{self.input.name:array})
            except Exception as exc:
                if self.dynamic_batch and self.batch>1 and any(token in str(exc).lower() for token in ("out of memory","failed to allocate","cuda failure 2")):
                    self.batch = max(1,self.batch//2)
                    continue
                raise
            for k,(i,j,_,center,scale) in enumerate(block):
                xy,confidence = postprocess([value[k:k+1] for value in values],np.array([256,256]),center,scale,2.0)
                points,scores=np.asarray(xy[0]),np.asarray(confidence[0])
                finite=np.isfinite(points).all(axis=1)&np.isfinite(scores)
                outputs[i][j].update(joints_2d=np.where(np.isfinite(points),points,0).tolist(),confidence=np.where(finite,scores,0).tolist(),invalid_points=int((~finite).sum()))
            offset += actual_count
            batches += 1
        return outputs,dict(seconds=time.monotonic()-start,providers=self.providers,batch=self.batch,batches=batches,crops=len(jobs))

def associate(reference_frames, detections):
    """Associate whole detected instances using boxes and temporal displacement."""
    from scipy.optimize import linear_sum_assignment
    output, previous = [], {}
    if len(reference_frames) != len(detections):
        raise ValueError("Hand association frame count mismatch")
    for ref, candidates in zip(reference_frames,detections):
        refs=[]
        for h in ref["hands"]:
            points=np.asarray(h["joints_2d"],float)
            valid=np.asarray(h["valid_2d"],bool)&np.isfinite(points).all(axis=1)
            in_frame=valid & (points[:,0]>=0) & (points[:,0]<ref['width']) & (points[:,1]>=0) & (points[:,1]<ref['height'])
            if in_frame.any():
                p=points[valid]
                refs.append((h["side"],np.concatenate([p.min(axis=0),p.max(axis=0)])))
        cost=np.zeros((len(refs),len(candidates)))
        scale=max(ref["width"],ref["height"])
        for i,(side,box) in enumerate(refs):
            center=(box[:2]+box[2:])/2
            for j,pred in enumerate(candidates):
                p=np.asarray(pred["box"]);pc=(p[:2]+p[2:])/2
                motion=np.linalg.norm(pc-previous[side]["prediction"]-(center-previous[side]["reference"]))/scale if side in previous else 0
                cost[i,j]=.7*np.linalg.norm(pc-center)/scale+.3*(1-box_iou(box,p))+.15*motion
        hands=[]
        if cost.size:
            for i,j in zip(*linear_sum_assignment(cost)):
                side,box=refs[i];p=np.asarray(candidates[j]["box"]);center=(box[:2]+box[2:])/2;pc=(p[:2]+p[2:])/2
                if np.linalg.norm(pc-center)/scale > .5:
                    continue
                hand=dict(candidates[j],side=side)
                hands.append(hand)
                previous[side]=dict(reference=center,prediction=pc)
        output.append(dict(sample_index=ref["sample_index"],hands=hands))
    return output
