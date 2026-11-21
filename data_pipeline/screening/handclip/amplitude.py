"""Independent control axes. No RGB synthesis, normalization of pose, or stabilization."""
from math import isfinite

def number(row, k):
    v=row[k]
    if isinstance(v,bool) or not isinstance(v,(int,float)) or not isfinite(v) or v<0:
        raise ValueError(k)
    return v

def hands(m):
    sal=m['motion_salience']
    if sal['version']!='palm_salience_v1':raise ValueError('version')
    ids=sal['participating_hands']
    if not ids or any(type(i) is not int or i not in (0,1) for i in ids) or len(set(ids))!=len(ids):
        raise ValueError('participants')
    return [sal['per_hand'][str(i)] for i in ids]

def band_reason(m,b):
    try:
        for key,field in [('min_translation_m','anchor_translation_max_m'),('min_rotation_deg','anchor_rotation_max_deg')]:
            if number(m,field)+1e-12 < b[key]:return 'below_camera_'+field
        for key,field in [('max_palm','palm_extent_lengths'),('max_screen','screen_extent_diag')]:
            if b[key] is not None and any(number(h,field)>b[key]+1e-12 for h in hands(m)):
                return 'above_hand_'+field
    except (TypeError,KeyError,ValueError):return 'missing_or_invalid_band_evidence'
    return None

def preview_reasons(m,c):
    reasons=[]
    try:
        for field,key in [('anchor_rotation_max_deg','anchor_rotation_deg'),('anchor_translation_max_m','anchor_translation_m'),
                          ('angular_velocity_max_deg_s','angular_velocity_max_deg_s'),('angular_velocity_p95_deg_s','angular_velocity_p95_deg_s'),
                          ('translation_velocity_max_m_s','translation_velocity_max_m_s'),('translation_velocity_p95_m_s','translation_velocity_p95_m_s')]:
            if number(m,field)>c['camera'][key]+1e-10:reasons.append('camera_'+field)
        for k in ('max_anchor_displacement_diag','max_pair_displacement_diag'):
            if number(m['rgb'],k)>c['rgb'][k]+1e-12:reasons.append('background_'+k)
        ms=c['motion_salience']; rows=hands(m); passed=[]
        for h in rows:
            palm,screen,active,static,spikes=[number(h,k) for k in ('palm_extent_lengths','screen_extent_diag','active_fraction','longest_static_seconds','isolated_spikes')]
            if active>1:raise ValueError('active_fraction')
            if spikes:reasons.append('hand_spike')
            passed.append(palm>=ms['min_palm_extent_lengths'] and screen>=ms['min_screen_extent_diag'] and active>=ms['min_active_fraction'] and static<=ms['max_static_seconds'])
        if not any(passed):reasons.append('no_single_hand_meets_motion_floors')
        b=band_reason(m,c['amplitude_bands'])
        if b:reasons.append(b)
    except (TypeError,KeyError,ValueError):reasons.append('missing_or_invalid_metrics')
    return list(dict.fromkeys(reasons))
