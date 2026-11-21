"""Validate caption, frame evidence and operation provenance."""
import math,re,json
from pathlib import Path
TAX=json.loads(Path(__file__).with_name("taxonomy.json").read_text())
CODES={v["code"]:(f,v) for f in TAX["families"] for v in f["subtypes"]}
PRESERVE_POLICY = "legacy_prose_recaption_preserve_operation"
PATTERN=re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*")
def words(s):return len(PATTERN.findall(s))
def check_text(text,global_unit):
 if not isinstance(text,str):raise ValueError('caption is not text')
 low,high=(1,95) if global_unit else (1,120)
 n=words(text)
 if n<1:raise ValueError('caption has no English words')
 parts=[s for s in re.split(r'\n\s*\n',text.strip()) if s.strip()]
 if len(parts) not in ((1,2) if global_unit else (2,)):raise ValueError('wrong paragraph count')
 if re.search(r'(?im)^\s*(?:[-*]|\d+[.)]|(?:scene|actions|environment|objects|task|subtask)\s*:)',text):raise ValueError('headings or lists')
def previous_classification(row):
 operation, provenance = row.get("previous_operation"), row.get("previous_operation_provenance")
 if (not isinstance(operation, dict) or not operation.get("code") or
         not isinstance(provenance, dict) or not provenance):
  raise ValueError("Recaptioning requires the previously accepted operation and its provenance")
 return operation, provenance

def check_operation(c,row,n):
 op=c['operation'];code=op['code'];assert code in set(CODES)|{'UNKNOWN'}
 if c.get("contract", {}).get("caption_policy") == PRESERVE_POLICY:
  operation, provenance = previous_classification(row)
  if op != operation or c["operation_provenance"] != provenance:
   raise ValueError("Recaptioning changed the preserved classification or its provenance")
  return
 if code!='UNKNOWN':
  f,s=CODES[code];assert op['family_id']==f['id'] and op['family']==f['label_en'] and op['subtype']==s['label_en']
 assert op['version']==TAX['version'] and c['operation_provenance']['mode']=='new_visual_classification'
 assert all(type(i)==int and 0<=i<len(c['segments']) for i in c['operation_provenance']['context_segment_indices'])
 sample=sorted({round(k*(n-1)/4) for k in range(5)})
 assert op['evidence_clip_frames'] and all(type(k)==int and k in sample for k in op['evidence_clip_frames'])
 assert op['evidence_source_frames']==[row['source_frames'][0]+k for k in op['evidence_clip_frames']] and op['rationale'].strip()

def check_caption(c,row):
 assert c['id']==row['id'] and c['video_sha']==row['video_sha'] and c['source_frames']==row['source_frames'] and c['domain']==row['domain']
 check_operation(c,row,row['frames'])
 check_text(c['task_text'],True);n=row['frames'];start=row['source_frames'][0];parts=max(3,math.ceil(n/240))
 global_sample=sorted({round(k*(n-1)/4) for k in range(5)})
 assert c['task_sampled_clip_frames']==global_sample and c['task_evidence_clip_frames'] and all(type(k)==int and k in global_sample for k in c['task_evidence_clip_frames'])
 assert c['task_evidence_source_frames']==[start+k for k in c['task_evidence_clip_frames']]
 assert len(c['segments'])==parts
 for j,s in enumerate(c['segments']):
  a,b=j*n//parts,(j+1)*n//parts;sample=sorted({a+round(k*(b-a-1)/4) for k in range(5)})
  check_text(s['subtask_text'],False)
  assert s['clip_frames']==[a,b] and s['source_frames']==[start+a,start+b] and s['sampled_clip_frames']==sample
  assert s['evidence_frames'] and all(type(k)==int and k in sample for k in s['evidence_frames'])
