"""Re-screen original robot episodes against their OWN native length histogram.
No concatenation, no new duration cap. All new files use isolated version roots.
"""
import argparse,collections,concurrent.futures as cf,fcntl,hashlib,json,os,subprocess,tarfile,tempfile,time,shutil
from pathlib import Path
import cv2
from robot_io import (
    ROBOT_VIEWS, absolute_record, allocate, atomic, bucket, case_id, encode, relative_record,
    sha, source_key, verify_clip_receipt, verify_ti5_source_receipt, write_jsonl,
)

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--domain',choices=['handson','ti5'],required=True)
p.add_argument('--source-root',type=Path,required=True)
p.add_argument('--root',type=Path,required=True)
p.add_argument('--holdout',type=Path,required=True)
p.add_argument('--inventory',type=Path,help='hands-on inventory snapshot matching the official manifest')
p.add_argument('--render-archive',type=Path,help='override the HandsOn render metadata archive')
p.add_argument('--workers',type=int,default=4)
p.add_argument('--target',type=int)
a=p.parse_args()
if a.workers<1 or (a.target is not None and a.target<1):
 p.error('target and worker count must be positive')
if a.domain=='handson' and a.inventory is None:
 p.error('--inventory is required for HandsOn')
root=a.root.resolve();root.mkdir(parents=True,exist_ok=True);(root/'clips').mkdir(exist_ok=True)
run_lock=(root/'.run.lock').open('a')
fcntl.flock(run_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
cv2.setNumThreads(1)
holdout=json.loads(a.holdout.read_text())
excluded=holdout['records'];raws=[];skip=collections.Counter()
contract=dict(domain=a.domain,target=a.target,source=a.source_root.name,
              holdout_sha256=sha(a.holdout),inventory_sha256=sha(a.inventory) if a.inventory else None)
contract_path=root/'screen_contract.json'
if contract_path.exists() and json.loads(contract_path.read_text())!=contract:
 raise ValueError('screening contract changed; use a new output directory')
complete_path = root / 'COMPLETE.json'
complete_path.unlink(missing_ok=True)
atomic(root / 'progress.json', dict(phase='reading_inventory', domain=a.domain, updated=time.time()))
atomic(contract_path,contract)
atomic(root/'exclusion500.json',holdout)
if a.domain=='handson':
 import sim_screen as screen
 source=a.source_root.resolve();snap=a.inventory
 official=[json.loads(s) for s in (source/'source_manifest.jsonl').read_text().splitlines() if s.strip()]
 enriched=[json.loads(s) for s in snap.read_text().splitlines() if s.strip()]
 inventory={r['episode_uid']:r for r in enriched}
 official_ids={r['episode_uid'] for r in official}
 if len(official_ids)!=len(official) or len(inventory)!=len(enriched) or official_ids!=set(inventory):
  raise ValueError('HandsOn manifest and inventory episode identities differ')
 excluded_ids={r['episode_uid'] for r in excluded if r.get('category')=='handson'};excluded_hash={r.get('raw_video_sha256') for r in excluded if r.get('category')=='handson'}
 for raw in official:
  uid=raw['episode_uid'];r=inventory[uid]
  for k in ['render_metadata_sha256','video_sha256','parquet_sha256','source_task','split']:
   if raw[k]!=r[k]:raise ValueError('source inventory changed '+uid)
  if uid in excluded_ids or r['video_sha256']['front'] in excluded_hash:skip['holdout500']+=1;continue
  if r['frames']<81:skip['under81']+=1;continue
  raws.append(dict(r,case_id='sim_'+uid[:20],source_video_path=str(source/r['video_paths']['front'])))
 archive_path=(a.render_archive or source/'provenance/render_metadata.tar').resolve()
 archive=tarfile.open(archive_path);members={m.name:m for m in archive if m.isfile()};target=a.target or 6000
else:
 import ti5_inventory as inventory_module
 import ti5_screen as screen
 rawroot=a.source_root.resolve()
 paths=sorted(rawroot.rglob('meta.json'))
 excluded_meta={r.get('source_meta') for r in excluded if r.get('category')=='real_robot'};excluded_hash={r.get('raw_video_sha256') for r in excluded if r.get('category')=='real_robot'}
 def parse(path):
  try:return dict(inventory_module.parse(path),source_episode_key=path.relative_to(rawroot).as_posix())
  except Exception as e:return dict(status='parse_error',error=str(e),source_metadata_path=str(path))
 atomic(root/'inventory_progress.json',dict(stage='reading_raw_metadata',paths=len(paths)))
 with cf.ThreadPoolExecutor(max_workers=a.workers) as pool:
  records=list(pool.map(parse,paths))
 write_jsonl(root/'inventory.all.jsonl',(relative_record(r,root) for r in records))
 for r in records:
  if not r.get('has_dexterous_hand_metadata'):skip[r['status']]+=1;continue
  if (r.get('source_metadata_path') in excluded_meta or str(Path(r['source_metadata_path']).relative_to(rawroot)) in excluded_meta):skip['holdout500']+=1;continue
  if not isinstance(r.get('frames'),int) or r['frames']<81:skip['under81_or_invalid']+=1;continue
  # Reference includes unreadable/missing candidates with valid raw length;
  # their unavoidable deficits are reported rather than silently renormalized.
  raws.append(r)
 target=a.target or 4000
hist=collections.Counter(bucket(r) for r in raws);quotas=allocate(hist,min(target,len(raws)));groups=collections.defaultdict(list)
for r in raws:groups[bucket(r)].append(r)
for q in groups.values():q.sort(key=lambda r:hashlib.sha256(source_key(r).encode()).digest())
atomic(root/'reference.json',dict(domain=a.domain,raw_reference_sources=len(raws),reference_histogram=dict(sorted(hist.items())),requested=target,initial_quotas=quotas,excluded=dict(skip),rule='one continuous prefix of each original episode, N=80*floor((raw_frames-1)/80)+1, minimum81, no added maximum',reference_before_quality=True,holdout_policy='exclude the frozen500 identities before normalization',native_source='official original episode lengths; no Ego reference reused'))
write_jsonl(root/'reference.inventory.jsonl',(relative_record(r,root) for r in raws))
selected = collections.defaultdict(list)
seen_hash, seen_raw = set(), set()
attempts = {}
ledger = root / 'qc_results.jsonl'
current_sources = {source_key(row): row for row in raws}
resume = collections.Counter()


def remember(receipt):
    attempts[receipt['key']] = receipt
    if receipt['status'] == 'pass':
        record = receipt['record']
        if record['video_sha'] not in seen_hash and record['raw_video_sha256'] not in seen_raw:
            selected[record['frames']].append(record)
            seen_hash.add(record['video_sha'])
            seen_raw.add(record['raw_video_sha256'])


if ledger.exists():
    # A repaired attempt appended later supersedes any older receipt for that source.
    with ledger.open() as journal:
        latest = {receipt['key']: receipt for receipt in map(json.loads, journal)}
    for key, receipt in latest.items():
        current = current_sources.get(key)
        if current is None:
            resume['not_in_current_inventory'] += 1
            continue
        if receipt['status'] != 'pass':
            continue
        try:
            record = absolute_record(receipt['record'], root)
            verify_clip_receipt(record, current, a.domain)
            if a.domain == 'handson':
                original = record['source_record']
                for field in ('video_sha256', 'parquet_sha256', 'render_metadata_sha256'):
                    if original[field] != current[field]:
                        raise ValueError('cached HandsOn source SHA-256 changed')
            else:
                verify_ti5_source_receipt(record, current)
            if record['raw_video_sha256'] in excluded_hash:
                raise ValueError('cached source is in the evaluation holdout')
        except (OSError, ValueError, KeyError, TypeError):
            resume['needs_recheck'] += 1
            continue
        remember(dict(receipt, record=record))
        resume['reused'] += 1
    for length, rows in groups.items():
        groups[length] = [row for row in rows if source_key(row) not in attempts]


def check(r):
 key=source_key(r);n=bucket(r);cid=case_id(a.domain,r);dest=root/'clips'/cid;dest.mkdir(exist_ok=True)
 try:
  with tempfile.TemporaryDirectory(prefix='robot_native_') as td:
   work=Path(td);fps=float(r['fps'])
   if a.domain=='handson':
    if abs(fps-30)>1e-5:raise ValueError('HandsOn source is not 30 FPS')
    if not r.get('render_verified') or not r.get('source_state_and_physics_validated') or r['render_qa'].get('black_or_flat_white_frames'):raise ValueError('source render/native validation failed')
    qc=screen.check_local(r,source,archive,members,work);views={v:encode(work/(v+'.mp4'),dest/(v+'.mp4'),n,fps) for v in ROBOT_VIEWS['handson']};gamma=1.
    native=dict(parquet=str(source/r['data_path']),parquet_sha256=r['parquet_sha256'],render_metadata_tar=str(archive_path),render_metadata_member=r['render_metadata_member'],render_metadata_sha256=r['render_metadata_sha256']);video=views['front'];raw_sha=r['video_sha256']['front']
   else:
    if r.get('status')!='metadata_candidate':raise ValueError('source metadata/files unavailable')
    if sha(r['source_metadata_path'])!=r['metadata_sha256']:raise ValueError('raw metadata changed')
    screen.stage(r['source_hdf5_path'],work/'native.hdf5');signals=screen.check_native(work/'native.hdf5',r['frames']);screen.stage(r['source_video_path'],work/'head.mp4');quality=screen.check_head(work/'head.mp4',r['frames'],fps)
    raw_sha=sha(work/'head.mp4')
    if raw_sha in excluded_hash:raise ValueError('frozen500 raw video SHA excluded')
    gamma=quality['planned_gamma']
    qc=dict(native_signals=signals,head_quality=quality);views=dict(head=encode(work/'head.mp4',dest/'head.mp4',n,fps,gamma));video=views['head'];native=dict(hdf5=r['source_hdf5_path'],hdf5_sha256=sha(work/'native.hdf5'),metadata=r['source_metadata_path'],metadata_sha256=r['metadata_sha256'])
   first_local=work/'first_frame.png';first=dest/'first_frame.png';subprocess.run(['ffmpeg','-v','error','-nostdin','-y','-threads','1','-filter_threads','1','-i',video['path'],'-frames:v','1','-threads','1',str(first_local)],check=True)
   with first_local.open('rb') as inc,first.open('wb') as out:shutil.copyfileobj(inc,out);out.flush();os.fsync(out.fileno())
   if sha(first)!=sha(first_local):raise IOError('first frame publish mismatch')
   record=dict(id=cid,raw_video_sha256=raw_sha,domain=a.domain,frames=n,fps=fps,source_frames=[0,n],source_raw_frames=r['frames'],source_identity=key,video=video['path'],video_sha=video['sha256'],videos=views,native_ref=native,first_frame=str(first),first_frame_sha256=sha(first),qc=qc,gamma=gamma,technical_review='pass',semantic_review='pending',human_review='pending',length_policy='own native original distribution; 80n+1 downward rounding; no new cap',source_record=r)
   atomic(dest/'clip.json',relative_record(record,dest))
   return dict(key=key,status='pass',frames=n,video_sha=video['sha256'],record=record)
 except Exception as e:return dict(key=key,status='failed',frames=n,error=type(e).__name__+': '+str(e))
def progress():
    counts = {length: len(rows) for length, rows in selected.items()}
    atomic(root / 'progress.json', dict(
        phase='complete' if complete_path.exists() else 'screening_and_export',
        domain=a.domain, attempted=len(attempts), selected=sum(counts.values()), target=target,
        by_length=counts, initial_quotas=quotas,
        failed=sum(row['status'] != 'pass' for row in attempts.values()),
        resume=dict(resume), human_review='pending', updated=time.time(),
    ))
progress()
with cf.ThreadPoolExecutor(max_workers=a.workers) as pool,ledger.open('a',buffering=1) as out:
 while True:
  pending=[];reserved=collections.Counter({k:len(v) for k,v in selected.items()})
  for _ in range(a.workers):
   buckets=[k for k,q in groups.items() if q and reserved[k]<quotas.get(k,0)]
   if not buckets:break
   k=max(buckets,key=lambda k:(quotas[k]-reserved[k],k));r=groups[k].pop()
   pending.append(r);reserved[k]+=1
  if not pending:break
  for v in pool.map(check,pending):
   receipt=dict(v,record=relative_record(v['record'],root)) if v['status']=='pass' else v
   out.write(json.dumps(receipt,ensure_ascii=False)+'\n');out.flush();os.fsync(out.fileno())
   remember(v)
  progress()
avail={k:len(v) for k,v in selected.items()};count=min(target,sum(avail.values()))
while count>0:
 q=allocate(hist,count)
 if all(q[k]<=avail.get(k,0) for k in q):break
 count-=1
q=allocate(hist,count);final=[r for k in sorted(selected) for r in selected[k][:q.get(k,0)]]
write_jsonl(root/'manifest.native_distribution.jsonl',(relative_record(r,root) for r in final))
technical=[r for clips in selected.values() for r in clips]
write_jsonl(root/'manifest.technical_passed.jsonl',(relative_record(r,root) for r in technical))
atomic(complete_path,dict(domain=a.domain,target=target,actual=len(final),deficit=target-len(final),histogram={k:q[k] for k in sorted(q)},raw_reference_histogram=dict(sorted(hist.items())),matching='Exact largest-remainder allocation at actual count; no silent substitution with Ego distribution',quality_passed_candidates=sum(avail.values()),initial_quota_shortfalls={k:quotas[k]-avail.get(k,0) for k in quotas if quotas[k]>avail.get(k,0)},manifest='manifest.native_distribution.jsonl',manifest_sha256=sha(root/'manifest.native_distribution.jsonl'),resume=dict(resume),human_review='pending',completed=time.time()))
progress()
