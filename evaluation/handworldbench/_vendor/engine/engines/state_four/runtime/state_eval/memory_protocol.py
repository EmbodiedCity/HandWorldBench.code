"""Opportunity gate and locator-specific independent facts."""
import json,re,math
from .io import sha
def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

def binary_score(answer, expected):
    if type(expected) is not bool:
        raise ValueError('Expected answer must be a boolean')
    choice = answer.get('choice') if answer.get('status') == 'ok' else None
    if choice not in ('1', '2'):
        return {'score': 0.0, 'raw_score': None, 'status': answer.get('status') if answer.get('status') != 'ok' else 'insufficient_evidence', 'invalid_as_zero': True}
    value = float((choice == '1') == expected)
    return {'score': value, 'raw_score': value, 'status': 'ok', 'invalid_as_zero': False}

def _sample(rows, count):
    # Uniform deterministic endpoints; no image quality / score selection.
    if len(rows) <= count:
        return rows
    return [rows[round(k * (len(rows) - 1) / (count - 1))] for k in range(count)]

def _frames(evidence, ids, count):
    if not ids:
        raise ValueError('Required window IDs are absent')
    rows = []
    for wid in ids:
        part = evidence['windows'].get(wid)
        if not part:
            raise ValueError('No mapped evidence in window ' + wid)
        rows.extend(part)
    seen = set()
    unique = []
    for row in sorted(rows, key=lambda x: float(x['source_time'])):
        key = (str(row['path']), float(row['source_time']))
        if key not in seen:
            unique.append(row)
            seen.add(key)
    return _sample(unique, count)

def _images(groups):
    from PIL import Image
    pictures = []
    try:
        for label, rows in groups:
            for k, row in enumerate(rows):
                if not row.get('sha256') or sha(row['path']) != row['sha256']:
                    raise ValueError('Changed or unhashed evidence image')
                with Image.open(row['path']) as image:
                    rgb = image.convert('RGB')
                pictures.append((label + ' ' + str(k + 1) + ' at source time ' + format(float(row['source_time']), '.4f') + ' seconds', rgb))
        return pictures
    except Exception:
        for _, image in pictures:
            image.close()
        raise

def _prompt(unit, question, mode, hidden=False):
    text = (
        'Evaluate a specific visual memory check, comparing the HISTORY images with the later RETURN images. '
        'These are observations from one video, not examples of success. '
        'HISTORY, INTERVENING and RETURN are intended temporal windows, not claims that the target is visible, hidden or successfully returned. '
        'Judge only the named historical detail in the same previously visible object region. '
        'Do not score general aesthetics, hand anatomy, overall task completion, or similarity to an imagined reference image. '
        'Viewpoint, scale, lighting and legitimate articulation may change. '
        'Use the actually shown earlier appearance as the comparison, not the description as proof. '
        'Do not assume persistence merely because the object category is unchanged. '
        'Do not invent details hidden by hands, blur or crops.\n'
        'Target detail: ' + unit['invariant'] + '\n'
        'Permitted changes: ' + '; '.join(unit.get('allowed_changes', [])) + '\n'
        'Question: ' + question + '\n'
    )
    if hidden:
        text += (
            'INTERVENING images cover the intended intervening opportunity; verify what actually happens. '
            'For this return gate, the target region must be visible in HISTORY, become unavailable '
            'through occlusion or turning away in the intervening images, and be visible again in RETURN. '
            'An unchanged continuously visible frozen view does not establish this required return. '
            'Visibility of the target region is enough for this gate: do not require its fine detail to be correct here.\n'
        )
    if mode == 'binary':
        text += 'Output exactly one digit: 1 = Yes; 2 = No; 3 = Cannot determine from the supplied evidence. No explanation.'
    else:
        text += (
            'Rate how strongly the visible compared detail satisfies the proposition in the question. '
            'This scale measures observed agreement, not confidence or general video quality. '
            '1 = clearly false / detail contradicted; '
            '2 = mostly false / major contradictions with limited matching detail; '
            '3 = mixed / substantive matching and conflicting detail; '
            '4 = mostly true / matching detail with minor localized discrepancies; '
            '5 = true / the requested coarse detail is preserved within the permitted changes. '
            '0 = the required comparison cannot be determined from the supplied images. '
            'Use 3 only for visible mixed evidence, not for unreadability. '
            'Blur preventing the specified coarse comparison warrants 0, not an invented match. '
            'Output exactly one digit and no explanation.'
        )
    return text

# Memory-consistency fact prompt: one codebook question about one localized feature, asked separately for the
# history and the return frames (the two answers are compared by score_pair).
def prompt_for(checklist,check):
 book=check['codebook'];options=[]
 for k in sorted(book,key=int):
  value='unknown or cannot be determined' if k=='0' else str(book[k]);options.append(k+' = '+value)
 lines=['Observation scope: '+checklist['feature_scope'],'Target local surface: '+checklist['surface_locator'],'Object-intrinsic coordinate frame for this item: '+json.dumps(check['coordinate_frame'],ensure_ascii=False),'Bucket definition: '+check['execution_definition'],'Question: '+check['question']]
 if check.get('roles'):lines.append('Role definitions: '+json.dumps(check['roles'],ensure_ascii=False))
 lines.append('Codebook: '+'; '.join(options))
 lines.append('These images come from the same observation window. Answer only from what is currently visible and do not add up counts across images. If the target, the property or the reference cannot be determined, or the images do not support one stable answer, answer 0. Output only one integer code, with no explanation.')
 return 'Object locator (only for finding the target): '+checklist['object_locator']+'\nFeature measured by this question: '+check['feature_target']+'\nMeasurement level: '+check['feature_target_level']+'\nThe object description is not the property to be answered. Answer only about the measured feature above; do not substitute the shape, color or count of the object carrying it or of the background. If the feature has clearly disappeared, use the absence code; if you cannot tell whether it is visible, use 0.\n'+'\n'.join(lines)

def validate_checklist(d):
 if not d.get('clip_id') or not d.get('checks'):raise ValueError('Missing clip/checks')
 ids=set()
 for c in d['checks']:
  if not c.get('id') or c['id'] in ids or not isinstance(c.get('question'),str) or not c['question'].strip():raise ValueError('Invalid check ID/question')
  ids.add(c['id']);b=c['codebook']
  if not c.get('feature_target') or c.get('feature_target_level') not in ('local_feature','feature_group','feature_relation','component','local_surface'):raise ValueError('Missing explicit measured feature and level')
  if not isinstance(c.get('execution_definition'),str) or not c['execution_definition'].strip():raise ValueError('Missing frozen bucket execution definition')
  frame=c.get('coordinate_frame',{})
  if not isinstance(frame,dict) or frame.get('type') not in ('directed','undirected','topological','not_applicable') or not isinstance(frame.get('definition'),str) or not frame['definition'].strip():raise ValueError('Missing explicit object coordinate frame')
  if c.get('family')=='spatial_structure' and frame['type']=='not_applicable':raise ValueError('Spatial property requires explicit intrinsic or topological frame')
  if frame['type']=='directed' and any(not isinstance(frame.get(k),str) or not frame[k].strip() for k in ('anchor','positive_direction')):raise ValueError('Directed coordinate frame needs anchor and polarity')
  absence=c.get('absence_codes',[])
  if not isinstance(absence,list) or any(type(x) is not int or x==0 or str(x) not in b for x in absence):raise ValueError('Invalid absence codes')
  if not isinstance(b,dict) or b.get('0','missing') is not None or len(b)<2 or any(not re.fullmatch(r'0|[1-9][0-9]*',k) for k in b):raise ValueError('Invalid frozen codebook')
  if any(v is None for k,v in b.items() if k!='0'):raise ValueError('Only0 may be unknown')
 for k in ('feature_scope','surface_locator','reference_frame','window_sources'):
  if not d.get(k):raise ValueError('Missing '+k)
 return True

def decoded(r,book):
 code=r.get('choice')
 if r.get('status')!='ok' or type(code) is not int or str(code) not in book:return {'code':None,'value':None,'unknown':False,'error':True}
 return {'code':code,'value':book[str(code)],'unknown':code==0,'error':False}

def score_pair(h,r,book,absence_codes=()):
 a=decoded(h,book);b=decoded(r,book);known=not(a['unknown'] or b['unknown'] or a['error'] or b['error'])
 return {'history':a,'return':b,'score':float(known and a['value']==b['value']),'known_pair':known,'both_absent':known and a['code'] in absence_codes and b['code'] in absence_codes,'unknown_observations':int(a['unknown'])+int(b['unknown']),'error_observations':int(a['error'])+int(b['error'])}

def validate_frames(frames):
 if not frames or len(frames)>3:raise ValueError('Need1–3 actual frames in each stage')
 prev=None
 for f in frames:
  if sha(f['path'])!=f['sha256']:raise ValueError('Frame pixel mismatch')
  t=f['source_time']
  if type(t) not in (int,float) or not math.isfinite(t) or t<0 or (prev is not None and t<=prev):raise ValueError('Invalid frame order')
  prev=t
 return True
import hashlib
