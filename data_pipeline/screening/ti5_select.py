"""Assemble the TI5 v4 selection with a soft RGB camera ranking."""
import argparse,collections,concurrent.futures as cf,fcntl,hashlib,json,os,tempfile,time
from pathlib import Path
import h5py
from robot_io import (
    absolute_record, allocate, atomic, bucket, case_id, encode, relative_record, sha, source_key,
    verify_clip_receipt, verify_ti5_source_receipt, write_jsonl as write,
)

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--base',type=Path,required=True,help='TI5 technical screening run')
p.add_argument('--proxy',type=Path,required=True,help='TI5 RGB proxy run')
p.add_argument('--out',type=Path,required=True)
p.add_argument('--target',type=int,default=4000)
p.add_argument('--workers',type=int,default=4)
a=p.parse_args()
if a.target<1 or a.workers<1:p.error('target and worker count must be positive')
root=a.out.resolve();root.mkdir(parents=True,exist_ok=True)
run_lock=(root/'.run.lock').open('a')
fcntl.flock(run_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
base=a.base.resolve();proxy=a.proxy.resolve()
inventory=base/'reference.inventory.jsonl'
config=json.loads((proxy/'config.json').read_text())
if config['inventory_sha256']!=sha(inventory):raise ValueError('proxy inventory and technical inventory differ')
raws=[absolute_record(r,base) for r in map(json.loads,inventory.read_text().splitlines())]
byid={case_id('ti5',r):r for r in raws}
if len(raws)!=len(byid):raise ValueError('duplicate TI5 source identity in reference inventory')
baseline={r['id']:absolute_record(r,base) for r in map(json.loads,(base/'manifest.technical_passed.jsonl').read_text().splitlines())}
scans={r['id']:r for r in (json.loads(file.read_text()) for file in (proxy/'scan').glob('*.json'))}
if set(scans)!=set(byid):raise ValueError('proxy receipts do not cover exactly the TI5 source inventory')
if any(scans[cid]['key']!=source_key(raw) or scans[cid]['frames']!=bucket(raw) for cid,raw in byid.items()):
 raise ValueError('proxy receipt source identity or interval changed')
thresholds=config['thresholds'];keys=list(thresholds)
reference=json.loads((base/'reference.json').read_text())
hist={int(k):v for k,v in reference['reference_histogram'].items()};target=a.target;quotas=allocate(hist,target)
excluded=json.loads((base/'exclusion500.json').read_text())['records']
excluded_meta={r.get('source_meta') for r in excluded if r.get('category')=='real_robot'}
excluded_hash={r.get('raw_video_sha256') for r in excluded if r.get('category')=='real_robot'}
(root/'clips').mkdir(exist_ok=True);(root/'asset_receipts').mkdir(exist_ok=True)

def attach(row,scan):
 return dict(row,camera_pose_measured=False,camera_proxy=scan,human_review='pending')

def export_candidate(cid, r, scan):
 n=bucket(r);dest=root/'clips'/cid;dest.mkdir(exist_ok=True)
 old=baseline.get(cid)
 if old is not None:
  try:
   verify_ti5_source_receipt(old,r)
  except (OSError,ValueError,KeyError,TypeError):
   old=None
  else:
   try:
    verify_clip_receipt(old,r,'ti5')
   except (OSError,ValueError,KeyError,TypeError):
    pass
   else:
    return attach(old,scan)

 if sha(r['source_metadata_path'])!=r['metadata_sha256']:
  raise ValueError('raw metadata changed after screening')
 # Preserve native validity masks for supplemental selections.
 native=dict(hdf5=r['source_hdf5_path'],metadata=r['source_metadata_path'],metadata_sha256=r['metadata_sha256'],source_slice=[0,n]);validity={}
 try:
  native['hdf5_sha256']=sha(r['source_hdf5_path'])
  with h5py.File(r['source_hdf5_path'],'r') as h:
   for k in ['invalid_value_fallback','invalid_observation_value_fallback','invalid_action_value_fallback']:
    key='synchronization/sample_quality/'+k;v=h.get(key);validity[k]=dict(path=key,available=v is not None,invalid_frames=int(v[:n].sum()) if v is not None and v.shape[0]>=n else None)
 except Exception as e:validity=dict(hdf5_readable=False,error=str(e))
 with tempfile.TemporaryDirectory(prefix='camera_relaxed_') as td:
  td=Path(td);source=td/'raw.mp4'
  with Path(r['source_video_path']).open('rb') as inc,source.open('wb') as out:
   import shutil;shutil.copyfileobj(inc,out,4<<20)
  rawsha=sha(source)
  if scan.get('raw_video_sha256') and rawsha!=scan['raw_video_sha256']:
   raise ValueError('raw video changed after the proxy scan')
  gamma=old.get('gamma',1.) if old is not None and old['raw_video_sha256']==rawsha else 1.
  video=None
  for _ in range(3):
   try:video=encode(source,dest/'head.mp4',n,float(r['fps']),gamma);break
   except Exception:time.sleep(1)
  if video is None:raise IOError('supplemental encode/publish failed')
  ff=td/'first.png';import subprocess
  subprocess.run(['ffmpeg','-v','error','-nostdin','-y','-threads','1','-filter_threads','1','-i',video['path'],'-frames:v','1','-threads','1',str(ff)],check=True)
  first=dest/'first_frame.png'
  with ff.open('rb') as inc,first.open('wb') as out:shutil.copyfileobj(inc,out);out.flush();os.fsync(out.fileno())
  if sha(ff)!=sha(first):raise IOError('first frame publish mismatch')
 record=dict(old or {})
 record.update(
     id=cid, domain='ti5', frames=n, fps=r['fps'], video=video['path'],
     video_sha=video['sha256'], videos={'head': video},
     source_frames=[0,n], source_raw_frames=r['frames'], source_identity=source_key(r),
     source_record=r, native_ref=native, native_validity=validity, raw_video_sha256=rawsha,
     first_frame=str(first), first_frame_sha256=sha(first),
     technical_review='camera_proxy_and_decode_pass; native_controls_masked_or_uncertified',
     semantic_review='pending', human_review='pending', gamma=gamma,
 )
 return attach(record,scan)

def assess(cid):
 scan=scans[cid];stats=scan.get('stats')
 if stats:
  score=max(stats[k]/thresholds[k] for k in keys)
  tier=0 if score<=1 else 2;confidence='medium_rgb_proxy' if tier==0 else 'quota_relaxed_motion_proxy'
 else:score=None;tier=1;confidence='low_insufficient_proxy_evidence'
 return tier,score,confidence
def rank(cid):
 tier,score,confidence=assess(cid)
 return (tier,cid not in baseline,score or 0,hashlib.sha256(cid.encode()).hexdigest())
groups=collections.defaultdict(list)
for cid,r in byid.items():
 if r['source_metadata_path'] in excluded_meta or source_key(r) in excluded_meta:continue
 if r.get('status')!='metadata_candidate':continue
 groups[(r['frames']-1)//80*80+1].append(cid)
for ids in groups.values():ids.sort(key=rank)
config=dict(target=target,policy='Quantity-first native length quotas; camera proxy SOFT ranking only, target fills native length quotas with explicit proxy confidence',moderate_reference_thresholds=thresholds,camera_hard_thresholds=False,missing_camera_evidence_admitted=True,measured_camera_pose=False,native_invalid_masks_preserved=True,source_reference_sha256=sha(base/'reference.json'),proxy_config_sha256=sha(proxy/'config.json'),proxy_receipts_sha256=hashlib.sha256(json.dumps(scans,sort_keys=True).encode()).hexdigest())
config_path=root/'config.json'
if config_path.exists() and json.loads(config_path.read_text())!=config:
 raise ValueError('TI5 selection input changed; use a new output directory')
complete_path=root/'COMPLETE.json'
complete_path.unlink(missing_ok=True)
atomic(config_path,config)
atomic(root/'progress.json',dict(phase='preparing',prepared=0,updated=time.time()))
ready={};blocked=json.loads((root/'SOURCE_INTEGRITY_EXCLUSIONS.json').read_text()) if (root/'SOURCE_INTEGRITY_EXCLUSIONS.json').exists() else {};seen_raw={};seen_video={}


def verify_selected_source(raw, scan, rawsha):
    if raw['source_metadata_path'] in excluded_meta or source_key(raw) in excluded_meta:
        raise ValueError('frozen500 metadata excluded')
    if scan.get('raw_video_sha256') and scan['raw_video_sha256'] != rawsha:
        raise ValueError('proxy and selected media refer to different raw videos')
    if rawsha in excluded_hash:
        raise ValueError('frozen500 raw SHA excluded')


def prepare(cid):
 path=root/'asset_receipts'/(cid+'.json')
 raw=byid[cid];scan=scans[cid]
 if path.exists():
  try:
   cached=absolute_record(json.loads(path.read_text()),root)
   verify_clip_receipt(cached,raw,'ti5')
   verify_ti5_source_receipt(cached,raw)
   verify_selected_source(raw,scan,cached['raw_video_sha256'])
  except (OSError,ValueError,KeyError,TypeError):
   pass
  else:
   return cached
 record=None
 for attempt in range(3):
  try:record=export_candidate(cid,raw,scan);break
  except Exception as e:
   if attempt==2:raise
   time.sleep(1+attempt)
 tier,score,confidence=assess(cid)
 record=dict(record,selection_policy='TI5-v4 native length quotas with a soft camera rank and explicit proxy confidence',camera_qc_status=confidence,camera_proxy_relaxation_tier=tier,camera_proxy_normalized_score=score,camera_proxy_reference_thresholds=thresholds,camera_trajectory_verified=False,camera_pose_measured=False,quality_review_required=tier!=0,human_review='pending')
 record['technical_review']='video integrity and provenance accepted; camera soft-rank/uncertainty retained'
 rawsha=record.get('raw_video_sha256')
 if not rawsha:
  # Re-read source bytes for unambiguous raw duplicate/holdout isolation.
  rawsha=sha(raw['source_video_path'])
 verify_selected_source(raw,scan,rawsha)
 record['raw_video_sha256']=rawsha
 atomic(path,relative_record(record,root));return record
# Fill native length quotas and redistribute shortages across available bins.
for round_index in range(6):
 available={k:sum(cid not in blocked for cid in ids) for k,ids in groups.items()};counts={k:min(quotas.get(k,0),available.get(k,0)) for k in hist}
 for _ in range(target-sum(counts.values())):
  choices=[k for k in hist if counts[k]<available.get(k,0)]
  if not choices:raise RuntimeError('fewer unique eligible clips than requested')
  k=min(choices,key=lambda k:((counts[k]+1)/target-hist[k]/sum(hist.values()),k));counts[k]+=1
 chosen=[cid for k in sorted(groups) for cid in [c for c in groups[k] if c not in blocked][:counts.get(k,0)]]
 assert len(chosen)==target
 pending=[c for c in chosen if c not in ready]
 def safe(cid):
  try:return cid,prepare(cid),None
  except Exception as e:return cid,None,type(e).__name__+': '+str(e)
 with cf.ThreadPoolExecutor(max_workers=a.workers) as pool:
  for cid,r,error in pool.map(safe,pending):
   if error:blocked[cid]=error
   else:ready[cid]=r
   if (len(ready)+len(blocked))%24==0:atomic(root/'progress.json',dict(phase='preparing',prepared=len(ready),blocked=len(blocked),selected=len(chosen),round=round_index,updated=time.time()))
 # Replace clips that repeat source or encoded bytes.
 seen_raw={};seen_video={}
 for cid in chosen:
  if cid not in ready:continue
  r=ready[cid]
  if r['raw_video_sha256'] in seen_raw or r['video_sha'] in seen_video:blocked[cid]='exact raw/video duplicate'
  else:seen_raw[r['raw_video_sha256']]=cid;seen_video[r['video_sha']]=cid
 if all(cid not in blocked and cid in ready for cid in chosen):break
else:raise RuntimeError('could not fill target after six replacement rounds')
final=[ready[cid] for cid in chosen];assert len(final)==target and len({r['id'] for r in final})==target
assert len({r['video_sha'] for r in final})==target and len({r['raw_video_sha256'] for r in final})==target
for r in final:assert r['frames']>=81 and (r['frames']-1)%80==0 and r['source_frames']==[0,r['frames']]
manifest='manifest.ti5.jsonl'
write(root/manifest,(relative_record(row,root) for row in final))
selected_hist=collections.Counter(r['frames'] for r in final);tiers=collections.Counter(r['camera_qc_status'] for r in final)
report=dict(actual=target,target=target,manifest=manifest,sha256=sha(root/manifest),completed=time.time(),camera_hard_thresholds=False,camera_measured_pose=False,all_camera_verified=False,camera_status_counts=dict(tiers),length_histogram=dict(sorted(selected_hist.items())),native_reference_histogram=hist,quotas=quotas,length_exact=dict(selected_hist)=={k:v for k,v in quotas.items() if v},length_total_variation=.5*sum(abs(selected_hist.get(k,0)/target-v/sum(hist.values())) for k,v in hist.items()),native_mask_policy='Original invalid-value and synchronization masks retained; never mark all native controls as valid',source_ids_unique=True,raw_sha_unique=True,video_sha_unique=True,frozen500_overlap=0,blocked=blocked,human_review='pending')
atomic(complete_path,report);atomic(root/'progress.json',dict(phase='complete',actual=target,completed=report['completed']));print(json.dumps(report,ensure_ascii=False),flush=True)
