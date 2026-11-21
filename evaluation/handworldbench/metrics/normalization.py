"""Fixed positive metric scales independent of compared models."""
import math
from handworldbench.registry import METRICS

def normalize(metric,raw,already_normalized=False):
    if metric not in METRICS:raise ValueError('Unknown metric: '+metric)
    if raw is None:return None
    raw=float(raw)
    if not math.isfinite(raw):raise ValueError('Nonfinite score')
    if already_normalized:return max(0.,min(1.,raw))
    if metric=='musiq':raw/=100.
    elif metric=='jepa_similarity':raw=math.exp(-.4*max(raw,0.))
    elif metric=='hand_morphology':raw=1-raw
    elif metric=='photometric_consistency':raw=1/(1+max(raw,0)/.01)
    return max(0.,min(1.,raw))
