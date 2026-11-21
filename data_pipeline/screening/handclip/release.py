"""Final event gate over an already selected and independently verified clip.

Caller supplies the current R5 selected pool, artifact and verification. This
gate cannot replace source decoding, source identity checks or global dedup.
It does not mutate an active production run or trust a legacy visual pass.
"""
import hashlib
import json
from pathlib import Path

from .config import config_hash
from .storage import read_json, _read_visible_bytes


def _bound_file(run, descriptor):
    """Read an immutable, hash-bound evidence file confined to this run."""
    if not isinstance(descriptor,dict):
        raise ValueError('missing evidence descriptor')
    name,digest=descriptor.get('path'),descriptor.get('sha256')
    if not isinstance(name,str) or not name or not isinstance(digest,str):
        raise ValueError('invalid evidence descriptor')
    root=Path(run).resolve()
    path=(root/name).resolve()
    if path==root or root not in path.parents:
        raise ValueError('evidence path outside run')
    payload=_read_visible_bytes(path)
    if hashlib.sha256(payload).hexdigest()!=digest:
        raise ValueError('evidence hash mismatch')
    return payload


def load_release_policy(run, cfg):
    """Legacy policy presence requires complete-event evidence, even if draft.

    An incomplete policy closes final release but does not stop candidate scans.
    Calibration evidence is explicit; this function never manufactures labels.
    """
    path=Path(run)/'release_policy.json'
    try:
        path.stat()
    except FileNotFoundError:
        # A new run cannot become a legacy run by losing its policy. The
        # explicit marker is preferred; the frozen release module also detects
        # runs created by the first event-aware version before that marker.
        manifest_path=Path(run)/'run.json'
        try:
            manifest_path.stat()
        except FileNotFoundError:
            manifest={}
        except OSError:
            return dict(required=True,reasons=['run_manifest_unreadable'],policy={},policy_sha256=None)
        else:
            try:
                manifest=read_json(manifest_path)
                if not isinstance(manifest,dict) or not isinstance(manifest.get('code_manifest',{}),dict):
                    raise ValueError('invalid run manifest')
            except (OSError,ValueError,TypeError):
                return dict(required=True,reasons=['run_manifest_unreadable'],policy={},policy_sha256=None)
        code_paths={str(name).replace('\\','/') for name in manifest.get('code_manifest',{})}
        if 'required_complete_atomic_event' in manifest or 'handclip/release.py' in code_paths:
            return dict(required=True,reasons=['release_policy_missing_for_run'],policy={},policy_sha256=None)
        return dict(required=False,reasons=[],policy=None,policy_sha256=None)
    except OSError:
        return dict(required=True,reasons=['release_policy_unreadable'],policy={},policy_sha256=None)
    payload=None
    try:
        payload=_read_visible_bytes(path)
        policy=json.loads(payload)
    except (OSError,ValueError,TypeError):
        return dict(required=True,reasons=['invalid_release_policy'],policy={},
                    policy_sha256=hashlib.sha256(payload).hexdigest() if payload is not None else None)
    reasons=[]
    if not isinstance(policy,dict):
        return dict(required=True,reasons=['invalid_release_policy'],policy={},
                    policy_sha256=hashlib.sha256(payload).hexdigest())
    if policy.get('required_complete_atomic_event') is not True:
        reasons.append('complete_event_requirement_missing')
    for key in ('minimum_pre_context_frames','minimum_post_context_frames'):
        if type(policy.get(key)) is not int or policy[key] <= 0:
            reasons.append('invalid_policy_'+key)
    if policy.get('config_sha256')!=config_hash(cfg):reasons.append('release_policy_config_mismatch')
    definition=policy.get('event_definition_file')
    try:
        _bound_file(run,definition)
    except (OSError,ValueError):reasons.append('event_definition_not_bound')
    try:
        calibration=json.loads(_bound_file(run,policy.get('calibration_file')))
        if not isinstance(calibration,dict):raise ValueError('invalid calibration')
        if calibration.get('status')!='validated' or calibration.get('config_sha256')!=config_hash(cfg):
            reasons.append('calibration_not_validated_for_config')
        dev=calibration.get('development_recording_ids')
        holdout=calibration.get('holdout_recording_ids')
        if (not isinstance(dev,list) or not isinstance(holdout,list) or not dev or not holdout
                or any(not isinstance(v,str) or not v for v in dev+holdout)
                or set(dev)&set(holdout)):
            reasons.append('calibration_splits_not_disjoint')
        for key in ('camera_stability','first_frame_hand_clarity'):
            descriptor=calibration.get('evidence',{}).get(key)
            _bound_file(run,descriptor)
    except (OSError,ValueError,TypeError,AttributeError):reasons.append('calibration_evidence_missing_or_invalid')
    return dict(required=True,reasons=reasons,policy=policy,
                policy_sha256=hashlib.sha256(payload).hexdigest())


def gate_review(run, cfg, artifact, verification, review, selected_ids, policy_state=None):
    """Additional gate; caller still applies R5's current selection and hashes."""
    state=policy_state if policy_state is not None else load_release_policy(run,cfg)
    if not state['required']:
        return dict(qualified=True,reasons=[],scope='legacy_run_without_event_policy')
    reasons=list(state['reasons'])
    try:
        payload=_bound_file(run,review.get('event_review_file'))
        policy=state['policy']
        result=event_release_gate(artifact,verification,payload,
            selected_clip_ids=selected_ids,
            expected_review_sha256=review['event_review_file']['sha256'],
            expected_definition_sha256=(policy.get('event_definition_file') or {}).get('sha256'),
            frame_multiple=cfg['selection'].get('frame_multiple',81),
            frame_offset=cfg['selection'].get('frame_offset',0),
            min_frames=cfg['selection'].get('min_frames',81),
            minimum_pre_context_frames=policy.get('minimum_pre_context_frames'),
            minimum_post_context_frames=policy.get('minimum_post_context_frames'))
        reasons+=result['reasons']
    except (OSError,ValueError,TypeError,KeyError,AttributeError):
        reasons.append('event_review_missing_or_invalid')
    if review.get('full_video_viewed') is not True or review.get('first_frame_viewed') is not True:
        reasons.append('complete_visual_review_not_recorded')
    return dict(qualified=not reasons,reasons=reasons,
                release_policy_sha256=state['policy_sha256'])


def event_release_gate(artifact, verification, review_bytes, *, selected_clip_ids,
                       expected_review_sha256, expected_definition_sha256,
                       frame_multiple=81, min_frames=81, frame_offset=0,
                       minimum_pre_context_frames=None, minimum_post_context_frames=None):
    """Validate frozen context minima and the outer uncertainty envelope.

    Context minima must be explicitly supplied by the frozen policy, never
    inferred from a review. Boundary ranges are closed ranges of local frame
    boundary coordinates. Exact boundaries use [s,s] and [e,e]; the reviewed
    half-open event interval must equal [start_range[0],end_range[1]].
    """
    reasons=[]
    if type(frame_multiple) is not int or frame_multiple <= 0 or type(min_frames) is not int or min_frames <= 0:
        raise ValueError('Positive integer frame parameters required')
    if not isinstance(review_bytes,bytes):
        raise ValueError('Exact saved review bytes required')
    actual_review_sha=hashlib.sha256(review_bytes).hexdigest()
    if actual_review_sha != expected_review_sha256:
        return dict(qualified=False,reasons=['review_hash_mismatch'])
    try:
        review=json.loads(review_bytes)
    except (UnicodeDecodeError,json.JSONDecodeError):
        return dict(qualified=False,reasons=['invalid_review_json'])
    if not isinstance(review,dict):
        return dict(qualified=False,reasons=['invalid_review_json'])
    policy_context_valid=all(type(v) is int and v > 0 for v in
                             (minimum_pre_context_frames,minimum_post_context_frames))
    if not policy_context_valid:
        reasons.append('invalid_frozen_context_minimum')
    cid=artifact.get('clip_id')
    video_sha=artifact.get('checksums',{}).get('video.mp4')
    c=artifact.get('candidate',{})
    if cid not in selected_clip_ids: reasons.append('not_in_current_selected_pool')
    if artifact.get('status') != 'ok': reasons.append('export_not_ok')
    if verification.get('status') != 'ok' or verification.get('clip_id') != cid:
        reasons.append('independent_verification_not_ok')
    if not video_sha or verification.get('video_sha256') != video_sha or review.get('video_sha256') != video_sha or review.get('clip_id') != cid:
        reasons.append('video_or_clip_binding_mismatch')
    start,end,n=c.get('start_frame'),c.get('end_frame'),c.get('n_frames')
    if any(type(v) is not int for v in (start,end,n)) or start < 0 or end-start != n or n < min_frames or (n-frame_offset) % frame_multiple:
        reasons.append('invalid_frame_interval')
    coverage=review.get('frame_coverage')
    if not (isinstance(coverage,list) and len(coverage)==2 and
            all(type(v) is int for v in coverage) and coverage == [0,n]):
        reasons.append('incomplete_full_clip_review')
    if review.get('visual_quality_verdict') != 'pass': reasons.append('visual_quality_not_pass')
    if review.get('atomic_event_verdict') != 'complete': reasons.append('event_not_complete')
    if not expected_definition_sha256 or review.get('event_definition_sha256') != expected_definition_sha256:
        reasons.append('event_definition_not_frozen_version')
    interval=review.get('event_local_interval')
    pre,post=review.get('minimum_pre_context_frames'),review.get('minimum_post_context_frames')
    interval_valid=(isinstance(interval,list) and len(interval)==2 and all(type(v) is int for v in interval))
    context_valid=all(type(v) is int and v > 0 for v in (pre,post))
    if not context_valid:
        reasons.append('invalid_review_context_minimum')
    elif policy_context_valid and (pre < minimum_pre_context_frames or post < minimum_post_context_frames):
        reasons.append('review_context_below_frozen_minimum')
    begin_range=review.get('event_start_boundary_range')
    end_range=review.get('event_end_boundary_range')
    ranges_valid=all(isinstance(v,list) and len(v)==2 and all(type(x) is int for x in v)
                     for v in (begin_range,end_range))
    ranges_valid=(ranges_valid and type(n) is int and
                  0 <= begin_range[0] <= begin_range[1] < end_range[0] <= end_range[1] <= n)
    envelope=[begin_range[0],end_range[1]] if ranges_valid else None
    if not ranges_valid:
        reasons.append('invalid_event_boundary_ranges')
    elif not interval_valid or interval != envelope:
        reasons.append('event_interval_not_outer_envelope')
    effective_pre=max(pre,minimum_pre_context_frames) if context_valid and policy_context_valid else None
    effective_post=max(post,minimum_post_context_frames) if context_valid and policy_context_valid else None
    if (not ranges_valid or effective_pre is None or
            not (effective_pre <= envelope[0] < envelope[1] <= n-effective_post)):
        reasons.append('incomplete_event_envelope')
    offset=c.get('source_frame_offset')
    if ranges_valid and type(start) is int and start >= 0 and type(offset) is int and offset >= 0:
        expected_global=[offset+start+v for v in envelope]
        actual_global=review.get('event_global_interval')
        if (not isinstance(actual_global,list) or len(actual_global)!=2 or
                any(type(v) is not int for v in actual_global) or actual_global != expected_global):
            reasons.append('global_event_alignment_mismatch')
    else:
        reasons.append('unknown_global_alignment')
    for key in ('event_definition','before','transition','after','reviewer_type','method'):
        if not isinstance(review.get(key),str) or not review[key].strip(): reasons.append('missing_'+key)
    return dict(qualified=not reasons,reasons=reasons,review_sha256=actual_review_sha,
                event_local_outer_envelope=envelope,
                effective_pre_context_frames=effective_pre,effective_post_context_frames=effective_post,
                scope='event_gate_only_requires_current_source_selection_and_independent_verification')
