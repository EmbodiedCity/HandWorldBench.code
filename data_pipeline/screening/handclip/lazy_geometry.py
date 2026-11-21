"""Exact temporal gates evaluated on demand, without materializing all windows.

The index stores O(N) frame data and O(N) upper bounds. Constructing anchor
envelopes is at worst O(N**2), but never evaluates a window percentile. Actual
P95/activity checks happen only in validate/longest_valid; their caller decides
which starts remain useful after RGB checks and nonoverlap selection.
"""
import math

import numpy as np

from . import temporal


def _next_bad(mask):
    """First bad frame at or after each index, or len(mask)."""
    n = len(mask)
    if not n:
        return np.empty(0, dtype=np.int64)
    return np.minimum.accumulate(np.where(mask, n, np.arange(n))[::-1])[::-1]


class GeometryIndex:
    """Hard-envelope upper bounds plus the unchanged exact temporal metrics.

    upper_bounds[s] is a frame count, not an end index. Zero means no window
    of the configured minimum length can pass at s. A nonzero value is only
    a hard-constraint bound: it is NOT a percentile/activity approval.
    Returned candidates still require RGB quality and source identity checks.
    This mutable, per-episode object is intended for one worker, not threads.
    """

    def __init__(self, ep, cfg):
        selection = cfg.get('selection', {})
        camera = cfg.get('camera', {})
        hand = cfg.get('hand', {})
        self.controls = cfg.get('amplitude_bands', {})
        self.n = int(ep['n'])
        self.multiple = int(selection.get('frame_multiple', 81))
        self.offset = int(selection.get('frame_offset', 0))
        if not 0 <= self.offset < self.multiple:
            raise ValueError('invalid frame_offset')
        minimum = int(selection.get('min_frames', self.multiple))
        self.maximum = selection.get('max_frames')
        self.exact = selection.get('exact_frames')
        if self.n < 0 or self.multiple <= 0 or minimum <= 0:
            raise ValueError('nonnegative n and positive frame_multiple/min_frames required')
        self.minimum = ((minimum - self.offset + self.multiple - 1) // self.multiple) * self.multiple + self.offset
        if self.exact is not None:
            self.exact = int(self.exact)
            if self.exact < self.minimum or (self.exact - self.offset) % self.multiple:
                raise ValueError('exact_frames must be divisible by frame_multiple and >= min_frames')
            self.minimum = self.maximum = self.exact
        if self.maximum is not None:
            self.maximum = int(self.maximum)
            if self.maximum < self.minimum:
                raise ValueError('max_frames is smaller than min_frames')
        self.upper_bounds = np.zeros(self.n, dtype=np.int64)
        self.hands = [0, 1]
        self.inference = {}
        self._window_start = None
        self._motion_cache = None
        self._salience = None
        self.diagnostics = dict(
            algorithm='lazy_geometry_index_v1', source_frames=self.n,
            minimum_frames=self.minimum, scanned_starts=max(0, self.n-self.minimum+1),
            hard_legal_starts=0, validation_calls=0, validation_successes=0,
            percentile_lengths_evaluated=0, longest_queries=0,
            threshold_status='run_configuration_snapshot',
            bounds_are_not_quality_approvals=True)
        # Match the old short-source contract: no projection or pose is needed.
        if self.n < self.minimum:
            return
        if cfg.get('motion_salience') is not None:
            from .motion_salience import MotionIndex
            self._salience = MotionIndex(ep, cfg['motion_salience'])

        n = self.n
        p = np.asarray(ep['positions_m'], dtype=float)
        self.t = np.asarray(ep['vio_t'], dtype=float)
        q = np.asarray(ep['vio_q_wxyz'], dtype=float)
        ts = np.asarray(ep['times_ns'], dtype=np.int64)
        if p.shape != (n, 2, 21, 3) or self.t.shape != (n, 3) or q.shape != (n, 4) or ts.shape != (n,):
            raise ValueError('episode arrays do not match the declared frame count')
        uv, depth_valid = temporal._project_hands(ep)
        uv, depth_valid = np.asarray(uv, dtype=float), np.asarray(depth_valid, dtype=bool)
        if uv.shape != (n, 2, 21, 2) or depth_valid.shape != (n, 2, 21):
            raise ValueError('project_hands returned unexpected shapes')
        width, height = int(ep['width']), int(ep['height'])
        border = float(hand.get('border_fraction', .01))
        min_span = float(hand.get('min_span_width_fraction', .04))
        max_invalid = int(hand.get('max_invalid_frames', 0))
        if not 0 <= border < .5 or min_span < 0 or max_invalid < 0:
            raise ValueError('invalid hand gate configuration')
        hand_valid = np.asarray(ep['hand_valid'], dtype=bool)
        finite_hand = np.isfinite(p).all(axis=(2, 3)) & np.isfinite(uv).all(axis=(2, 3))
        inside = ((uv[..., 0] >= width*border) & (uv[..., 0] <= width*(1-border)-1)
                  & (uv[..., 1] >= height*border) & (uv[..., 1] <= height*(1-border)-1))
        self.span = (np.max(uv[..., 0], axis=2)-np.min(uv[..., 0], axis=2))/width
        complete = hand_valid & finite_hand & np.all(depth_valid & inside, axis=2) & (self.span >= min_span)
        visible_joints = (np.isfinite(uv).all(axis=-1) & depth_valid
                          & (uv[..., 0] >= 0) & (uv[..., 0] < width)
                          & (uv[..., 1] >= 0) & (uv[..., 1] < height))
        visible = hand_valid & np.any(visible_joints, axis=2)
        self.hands, self.inference = temporal._participants(ep, complete, hand, visible)
        hand_hard_ok = np.all(hand_valid[:, self.hands] & finite_hand[:, self.hands]
                             & np.all(depth_valid[:, self.hands], axis=2), axis=1)
        self.complete_frame = np.all(complete[:, self.hands], axis=1)
        self.diagnostics.update(hand_valid_fraction=np.mean(hand_valid & finite_hand, axis=0).tolist(),
                                hand_complete_fraction=np.mean(complete, axis=0).tolist())
        qnorm = np.linalg.norm(q, axis=1)
        pose_ok = (np.asarray(ep['pose_valid'], dtype=bool) & np.isfinite(self.t).all(axis=1)
                   & np.isfinite(q).all(axis=1) & np.isfinite(qnorm) & (np.abs(qnorm-1) <= .01))
        self.q = q / np.where(qnorm > 0, qnorm, np.nan)[:, None]
        dt = np.diff(ts).astype(float)/1e9
        self.velocity = temporal._quat_angles(self.q[1:], self.q[:-1])/np.where(dt > 0, dt, np.nan)
        self.translation_velocity = np.linalg.norm(np.diff(self.t, axis=0), axis=1)/np.where(dt > 0, dt, np.nan)
        self.world = np.einsum('nij,nhkj->nhki', temporal._rotation_matrices(self.q), p) + self.t[:, None, None, :]
        self.palm_lengths = np.linalg.norm(p[:, :, 9]-p[:, :, 0], axis=-1)
        self.intrinsic_lengths = np.linalg.norm(p[:, :, 1:]-p[:, :, :1], axis=-1)
        self.max_translation = float(camera.get('anchor_translation_m', .03))
        self.max_rotation = float(camera.get('anchor_rotation_deg', 2.0))
        self.p95_velocity = float(camera.get('angular_velocity_p95_deg_s', 8.0))
        max_velocity = float(camera.get('angular_velocity_max_deg_s', 15.0))
        self.p95_translation_velocity = float(camera.get('translation_velocity_p95_m_s', .10))
        max_translation_velocity = float(camera.get('translation_velocity_max_m_s', .20))
        self.min_motion = float(hand.get('min_motion_palm_lengths', .08))
        if (not np.isfinite(self.p95_translation_velocity) or not np.isfinite(max_translation_velocity)
                or min(self.p95_translation_velocity, max_translation_velocity) < 0):
            raise ValueError('translation velocity thresholds must be finite and nonnegative')
        if min(self.max_translation, self.max_rotation, self.p95_velocity, max_velocity, self.min_motion) < 0:
            raise ValueError('camera and activity thresholds cannot be negative')

        # These O(N) barriers are independent of the anchor. A bad edge at i
        # forbids crossing i -> i+1, but a new window may begin at i+1.
        static_end = _next_bad(pose_ok & hand_hard_ok)
        edge_ok = (np.isfinite(self.velocity) & (dt > 0) & (self.velocity <= max_velocity+1e-10)
                   & np.isfinite(self.translation_velocity)
                   & (self.translation_velocity <= max_translation_velocity+1e-12))
        edge_end = np.r_[_next_bad(edge_ok)+1, n]
        self.diagnostics['peak_angular_bad_edges'] = int(np.count_nonzero(self.velocity > max_velocity))
        self.diagnostics['peak_translation_bad_edges'] = int(np.count_nonzero(self.translation_velocity > max_translation_velocity))
        self.diagnostics['invalid_pose_frames'] = int(np.count_nonzero(~pose_ok))
        self.diagnostics['incomplete_hand_frames'] = int(np.count_nonzero(~self.complete_frame))
        invalid_at = np.flatnonzero(~self.complete_frame)
        invalid_rank = np.searchsorted(invalid_at, np.arange(n)) + max_invalid
        invalid_end = np.full(n, n, dtype=np.int64)
        available = invalid_rank < len(invalid_at)
        invalid_end[available] = invalid_at[invalid_rank[available]]
        limits = np.minimum(np.minimum(static_end, edge_end), invalid_end)
        if self.maximum is not None:
            limits = np.minimum(limits, np.arange(n)+self.maximum)
        for start in range(n-self.minimum+1):
            if not pose_ok[start] or not hand_hard_ok[start] or not self.complete_frame[start]:
                continue
            limit = int(limits[start])
            if limit-start < self.minimum:
                continue
            translation = np.linalg.norm(self.t[start:limit]-self.t[start], axis=1)
            angle = temporal._quat_angles(self.q[start:limit], self.q[start])
            good = (np.isfinite(translation) & (translation <= self.max_translation+1e-12)
                    & np.isfinite(angle) & (angle <= self.max_rotation+1e-10))
            for key, mask in [('anchor_translation_truncated_starts', translation > self.max_translation+1e-12), ('anchor_rotation_truncated_starts', angle > self.max_rotation+1e-10)]:
                if np.any(mask):self.diagnostics[key] = self.diagnostics.get(key,0)+1
            bad = np.flatnonzero(~good)
            length = int(bad[0]) if len(bad) else limit-start
            length = (length-self.offset)//self.multiple*self.multiple+self.offset
            if length >= self.minimum:
                self.upper_bounds[start] = length
        self.diagnostics['hard_legal_starts'] = int(np.count_nonzero(self.upper_bounds))
        self.diagnostics['maximum_hard_upper_bound'] = int(self.upper_bounds.max(initial=0))

    def validate(self, start, length):
        """Check exactly [start,start+length); return a temporal candidate or None."""
        self.diagnostics['validation_calls'] += 1
        if (isinstance(start, (bool, np.bool_)) or isinstance(length, (bool, np.bool_))
                or not isinstance(start, (int, np.integer)) or not isinstance(length, (int, np.integer))):
            return None
        start, length = int(start), int(length)
        if (not 0 <= start < self.n or length < self.minimum or (length-self.offset) % self.multiple
                or length > self.upper_bounds[start] or (self.exact is not None and length != self.exact)):
            return None
        end = start+length
        speeds = self.velocity[start:end-1]
        self.diagnostics['percentile_lengths_evaluated'] += 1
        angular_p95 = float(np.percentile(speeds, 95)) if len(speeds) else 0.0
        if angular_p95 > self.p95_velocity+1e-10:
            self.diagnostics['angular_p95_rejected_queries']=self.diagnostics.get('angular_p95_rejected_queries',0)+1
            return None
        translation_speeds = self.translation_velocity[start:end-1]
        translation_p95 = float(np.percentile(translation_speeds, 95)) if len(translation_speeds) else 0.0
        if not np.isfinite(translation_p95) or translation_p95 > self.p95_translation_velocity+1e-12:
            self.diagnostics['translation_p95_rejected_queries']=self.diagnostics.get('translation_p95_rejected_queries',0)+1
            return None
        if self._window_start != start:
            bound = start+int(self.upper_bounds[start])
            self._translation = np.linalg.norm(self.t[start:bound]-self.t[start], axis=1)
            self._angle = temporal._quat_angles(self.q[start:bound], self.q[start])
            self._motion_cache = temporal._motion_features(
                self.world, self.intrinsic_lengths, self.palm_lengths, start, bound, self.hands)
            self._window_start = start
        motion = temporal._motion_metrics(*self._motion_cache, length, self.hands)
        if motion is None or not np.isfinite(motion['motion_palm_lengths']) or motion['motion_palm_lengths'] < self.min_motion:
            return None
        salience = self._salience.evaluate(start, end, self.hands) if self._salience is not None else None
        if salience is not None and salience['status'] != 'pass':
            self.diagnostics['motion_salience_rejects'] = self.diagnostics.get('motion_salience_rejects', 0) + 1
            return None
        metrics = dict(
            anchor_translation_max_m=float(np.max(self._translation[:length])),
            anchor_rotation_max_deg=float(np.max(self._angle[:length])),
            angular_velocity_p95_deg_s=angular_p95,
            angular_velocity_max_deg_s=float(np.max(speeds)) if len(speeds) else 0.0,
            translation_velocity_p95_m_s=translation_p95,
            translation_velocity_max_m_s=float(np.max(translation_speeds)) if len(translation_speeds) else 0.0,
            hand_invalid_frames=int(np.count_nonzero(~self.complete_frame[start:end])),
            min_hand_span_width_fraction=float(np.min(self.span[start:end, self.hands])),
            **motion)
        if salience is not None:
            metrics['motion_salience'] = salience
        if self.controls:
            from .amplitude import band_reason
            reason = band_reason(metrics,self.controls)
            if reason:
                self.diagnostics[reason]=self.diagnostics.get(reason,0)+1
                return None
        self.diagnostics['validation_successes'] += 1
        return dict(start_frame=start, end_frame=end, n_frames=length,
                    participating_hands=list(self.hands), participant_inference=dict(self.inference),
                    metrics=metrics, qc_status='pending')

    def longest_valid(self, start, upper=None):
        """Try descending multiples exactly; P95/activity are not monotonic."""
        self.diagnostics['longest_queries'] += 1
        if isinstance(start, (bool, np.bool_)) or not isinstance(start, (int, np.integer)) or not 0 <= start < self.n:
            return None
        start = int(start)
        if upper is not None and (isinstance(upper, (bool, np.bool_)) or not isinstance(upper, (int, np.integer))):
            return None
        bound = int(self.upper_bounds[start])
        if upper is not None:
            bound = min(bound, int(upper))
        bound = (bound-self.offset)//self.multiple*self.multiple+self.offset
        for length in range(bound, self.minimum-1, -self.multiple):
            candidate = self.validate(start, length)
            if candidate is not None:
                return candidate
        return None
