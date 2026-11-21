"""Import committed analysis only when all analysis inputs/gates are identical."""
from .storage import read_json,read_jsonl,sha256_file,append_segment,atomic_json
from .length_contract import validate_interval
from pathlib import Path

def seed(run,cfg,files):
    reuse=cfg.get('reuse_analysis')
    if not reuse:return
    receipt=Path(run)/'analysis_reuse_receipt.json'
    if receipt.exists():
        if read_json(receipt)['snapshot_sha256']!=reuse['sha256']:raise ValueError('reuse identity changed')
        return
    parent=read_json(reuse['parent_config'])
    for key in ('camera','hand','rgb','motion_salience','selection','aesthetics'):
        if cfg[key]!=parent[key]:raise ValueError('analysis gate changed: '+key)
    if cfg['curation']['max_candidates_per_source']!=parent['curation']['max_candidates_per_source']:
        raise ValueError('source proposal cap changed')
    if cfg['expansion']['seed_candidates_sha256']!=parent['expansion']['seed_candidates_sha256']:
        raise ValueError('analysis exclusions changed')
    if sha256_file(reuse['path'])!=reuse['sha256']:raise ValueError('analysis snapshot changed')
    source=read_jsonl(reuse['path']);allowed=set(files)
    if len({r['source_path'] for r in source})!=len(source):raise ValueError('duplicate imported source')
    from .curate import validate_score
    for row in source:
        if row['source_path'] not in allowed or row['status']!='ok':raise ValueError('unverified imported source')
        for c in row.get('candidates',[]):
            validate_interval(c,cfg['selection'])
            if c['source_path']!=row['source_path'] or c['source_sha256_at_analysis']!=row['source_sha256']:
                raise ValueError('imported source hash mismatch')
            if c['metrics']['motion_salience']['status']!='pass':raise ValueError('imported motion not passed')
            validate_score(c['aesthetics'],c['aesthetic_samples']['source_frame_indices'])
    existing={r['source_path']:r for r in read_jsonl(Path(run)/'analysis.jsonl')}
    for row in source:
        if row['source_path'] in existing and existing[row['source_path']]!=row:raise ValueError('import conflicts with current ledger')
    missing=[r for r in source if r['source_path'] not in existing]
    if missing:append_segment(Path(run)/'analysis.jsonl',missing)
    atomic_json(receipt,dict(snapshot_sha256=reuse['sha256'],imported_sources=len(source),
        parent_run=reuse['parent_run'],quality_operators='same geometry/RGB/motion/model scores; only scheduling and selection changed',
        outputs='old videos retained; current selected clips independently exported and verified into new directory'))
