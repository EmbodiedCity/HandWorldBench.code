#!/usr/bin/env python
"""Render skeleton / surface / X-ray control videos for HandWorldBench cases.

Examples
--------
Benchmark, three cases::

    python render_controls.py --data-root data/HandWorldBench --out outputs/controls \
        --mano-dir third_party/mano --wuji-meshes third_party/wuji-hand-description/meshes \
        --cases hw500_0001 hw500_0323 hw500_0431

All 500 benchmark cases with 16 worker processes::

    python render_controls.py --data-root data/HandWorldBench --out outputs/controls \
        --mano-dir third_party/mano --wuji-meshes third_party/wuji-hand-description/meshes --all --workers 16

Outputs: ``<out>/{skeleton,surface,xray}/<case_id>.mp4`` and ``<out>/receipts/<case_id>.json``.
"""
import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

_STATE = {}


def _init(args):
    """Per-process setup: datasets, hand models."""
    import numpy as np  # noqa: F401
    from datasets import Dataset
    _STATE['args'] = args
    _STATE['ds'] = Dataset(args.data_root, args.layout)


def _robots():
    if 'robots' not in _STATE:
        from robot import Robot
        a = _STATE['args']
        if not a.wuji_meshes:
            raise ValueError('HandsOn cases need --wuji-meshes (wuji-hand-description/meshes)')
        _STATE['robots'] = [Robot(s, a.wuji_meshes) for s in ('left', 'right')]
    return _STATE['robots']


def _faces():
    if 'faces' not in _STATE:
        from render_case import mano_faces
        a = _STATE['args']
        if not a.mano_dir:
            raise ValueError('human cases need --mano-dir (folder with MANO_LEFT.pkl and MANO_RIGHT.pkl)')
        _STATE['faces'] = mano_faces(a.mano_dir)
    return _STATE['faces']


def outputs_exist(out, cid):
    return all((out / s / f'{cid}.mp4').is_file() for s in ('skeleton', 'surface', 'xray')) \
        and (out / 'receipts' / f'{cid}.json').is_file()


def render_one(cid):
    import numpy as np
    import render_case
    a = _STATE['args']
    out = Path(a.out)
    if not a.overwrite and outputs_exist(out, cid):
        return dict(case_id=cid, status='exists')
    t0 = time.monotonic()
    try:
        case = _STATE['ds'].load(cid)
        mano = robots = faces = None
        if case.embodiment == 'human':
            import mano_fit
            faces = _faces()
            mano = mano_fit.load_or_fit(Path(a.work_dir) / 'mano' / f'{cid}.npz', case.joints, case.hand_valid,
                                        a.mano_dir, a.fit_threads, log=None)
        elif case.embodiment == 'handson':
            from handson_ik import recover
            robots = _robots()
            q, T = [], []
            for s in range(2):
                qs, Ts, err = recover(robots[s], case.joints[:, s])
                use = case.hand_valid[:, s] & np.isfinite(case.joints[:, s]).all((1, 2))
                if use.any() and err[use].max() > 1e-6:
                    raise ValueError(f'{cid}: hand {s} joints are not reachable by the Wuji model '
                                     f'(residual {err[use].max():.2e} m)')
                q.append(qs)
                T.append(Ts)
            case.hand20 = np.stack(q, 1)
            case.palm_to_camera = np.stack(T, 1)
        rec = render_case.run(case, out, mano=mano, robots=robots, faces=faces, previews=a.previews,
                              tmp_root=a.tmp_dir)
        rec['seconds_total'] = round(time.monotonic() - t0, 2)
        (out / 'receipts').mkdir(parents=True, exist_ok=True)
        (out / 'receipts' / f'{cid}.json').write_text(json.dumps(rec, indent=1))
        return dict(case_id=cid, status='ok', seconds=rec['seconds_total'])
    except Exception:
        return dict(case_id=cid, status='failed', error=traceback.format_exc())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-root', required=True, type=Path, help='benchmark or corpus dataset root')
    p.add_argument('--layout', choices=['benchmark', 'corpus'], help='default: auto-detect')
    p.add_argument('--out', required=True, type=Path, help='output folder')
    sel = p.add_mutually_exclusive_group(required=True)
    sel.add_argument('--cases', nargs='+', help='case / clip ids')
    sel.add_argument('--cases-file', type=Path, help='text file, one id per line')
    sel.add_argument('--all', action='store_true', help='every case in the dataset')
    p.add_argument('--embodiment', choices=['human', 'handson', 'ti5'], help='only cases of this embodiment')
    p.add_argument('--mano-dir', type=Path, help='folder with MANO_LEFT.pkl / MANO_RIGHT.pkl (human cases)')
    p.add_argument('--wuji-meshes', type=Path, help='wuji-hand-description/meshes (HandsOn cases)')
    p.add_argument('--work-dir', type=Path, help='cache for MANO fits (default: <out>/work)')
    p.add_argument('--tmp-dir', type=Path, help='where videos are encoded before being copied to --out '
                                                '(default: $TMPDIR); use a local disk')
    p.add_argument('--workers', type=int, default=4, help='parallel worker processes (default 4)')
    p.add_argument('--fit-threads', type=int, default=6, help='CPU threads per MANO fit (default 6)')
    p.add_argument('--shard', type=int, default=0, help='this machine\'s shard index (with --num-shards)')
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--overwrite', action='store_true', help='re-render cases whose outputs exist')
    p.add_argument('--previews', action='store_true', help='also write overlay JPEGs to <out>/previews/<id>/')
    a = p.parse_args(argv)
    a.work_dir = a.work_dir or a.out / 'work'

    from datasets import Dataset, list_cases
    ds = Dataset(a.data_root, a.layout)
    a.layout = ds.layout
    known = dict(list_cases(a.data_root, ds.layout))
    if a.all:
        ids = list(known)
    else:
        ids = a.cases or [s.strip() for s in a.cases_file.read_text().splitlines() if s.strip()]
        missing = [i for i in ids if i not in known]
        if missing:
            p.error(f'unknown ids: {missing[:5]}')
    if a.embodiment:
        ids = [i for i in ids if known[i] == a.embodiment]
    ids = ids[a.shard::a.num_shards]
    print(json.dumps(dict(layout=ds.layout, cases=len(ids), workers=a.workers)), flush=True)

    failed = 0
    if a.workers <= 1:
        _init(a)
        results = map(render_one, ids)
        for r in results:
            failed += r['status'] == 'failed'
            print(json.dumps(r), flush=True)
    else:
        with cf.ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a,)) as pool:
            for f in cf.as_completed([pool.submit(render_one, i) for i in ids]):
                r = f.result()
                failed += r['status'] == 'failed'
                print(json.dumps(r), flush=True)
    print(json.dumps(dict(done=len(ids), failed=failed)), flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    os.environ.setdefault('OMP_NUM_THREADS', '6')
    sys.exit(main())
