"""Metric names, categories, protocols, and observation dependencies."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Metric:
    id: str
    name: str
    name_en: str
    category: str
    observations: tuple[str, ...]
    level: str = "video"
    protocol: str = "1"

CATEGORIES = {
    "visual_quality": ("Visual Quality", "Visual Quality"),
    "hand_fidelity": ("Hand Fidelity", "Hand Fidelity"),
    "appearance_consistency": ("Appearance Consistency", "Appearance Consistency"),
    "instruction_following": ("Instruction Following", "Instruction Following"),
    "physical_interaction": ("Physical Interaction", "Physical Interaction"),
    "temporal_coherence": ("Temporal Coherence", "Temporal Coherence"),
}
_METRICS = [
    Metric("musiq", "MUSIQ", "MUSIQ", "visual_quality", ("image_quality",)),
    Metric("visual_plausibility", "Visual Plausibility", "Visual Plausibility", "visual_quality", ("visual_judgment",)),
    Metric("jepa_similarity", "JEPA Similarity", "JEPA Similarity", "visual_quality", ("video_features",), "collection"),
    Metric("pose_control", "Pose Control", "Pose Control", "hand_fidelity", ("reference_geometry", "hand_keypoints"), protocol="capped_distance_21pt_v1"),
    Metric("keypoint_coverage", "Keypoint Coverage", "Keypoint Coverage", "hand_fidelity", ("reference_geometry", "hand_keypoints"), protocol="geometric_21pt_v1"),
    Metric("hand_morphology", "Hand Morphology", "Hand Morphology", "hand_fidelity", ("hand_anatomy",), protocol="shared_dino_v1"),
    Metric("entity_consistency", "Entity Consistency", "Entity Consistency", "appearance_consistency", ("entity_features",), protocol="entity_regions_in_frame_v1"),
    Metric("background_consistency", "Background Consistency", "Background Consistency", "appearance_consistency", ("background_features",)),
    Metric("photometric_consistency", "Photometric Consistency", "Photometric Consistency", "appearance_consistency", ("background_geometry",)),
    Metric("action_adherence", "Action Adherence", "Action Adherence", "instruction_following", ("action_evidence",)),
    Metric("event_adherence", "Event Adherence", "Event Adherence", "instruction_following", ("event_evidence",)),
    Metric("trajectory_accuracy", "Trajectory Accuracy", "Trajectory Accuracy", "physical_interaction", ("object_tracks",)),
    Metric("causal_plausibility", "Causal Plausibility", "Causal Plausibility", "physical_interaction", ("causal_evidence",)),
    Metric("evolution_correctness", "Evolution Correctness", "Evolution Correctness", "physical_interaction", ("state_evidence",), protocol="shared_dino_v1"),
    Metric("state_regulation", "State Regulation", "State Regulation", "physical_interaction", ("state_evidence",), protocol="state_regulation_legal_termination_v1"),
    Metric("state_progression", "State Progression", "State Progression", "temporal_coherence", ("state_evidence",), protocol="shared_dino_v1"),
    Metric("memory_consistency", "Memory Consistency", "Memory Consistency", "temporal_coherence", ("memory_evidence",), protocol="shared_dino_v1"),
]
METRICS = {metric.id: metric for metric in _METRICS}
DEPENDENCIES = {
    "frames": (), "reference_frames": ("frames",), "reference_geometry": ("frames", "reference_frames"),
    "hand_detection": ("frames", "reference_geometry"), "hand_keypoints": ("hand_detection",),
    "hand_masks": ("hand_detection",), "object_tracks": ("frames", "reference_frames"),
    "entity_features": ("reference_geometry", "hand_masks", "object_tracks"),
    **{name: ("frames",) for name in ("image_quality", "visual_judgment", "video_features", "hand_anatomy", "background_features", "background_geometry", "action_evidence", "event_evidence", "causal_evidence", "state_evidence", "memory_evidence")},
}

def select(value=None):
    result = list(METRICS) if value is None else list(dict.fromkeys(value.split(",") if isinstance(value, str) else value))
    unknown = set(result) - METRICS.keys()
    if unknown:
        raise ValueError(f"Unknown metrics: {', '.join(sorted(unknown))}. Available: {', '.join(METRICS)}")
    return result

def observation_order(metrics):
    result = []
    def visit(name):
        if name in result:
            return
        for dependency in DEPENDENCIES[name]:
            visit(dependency)
        result.append(name)
    for metric in select(metrics):
        for name in METRICS[metric].observations:
            visit(name)
    return result


DEPENDENCIES['hand_anatomy']=('frames','reference_geometry','hand_detection')
