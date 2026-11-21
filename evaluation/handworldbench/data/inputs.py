"""Normalize a single model against a dataset catalog."""
from pathlib import Path
from handworldbench.registry import select
from handworldbench.runtime.storage import read, file_digest
from handworldbench.data.submission import complete_mappings, validate_automatic_protocol

ASSET_FIELDS = {'video', 'path', 'semantic', 'annotation', 'memory_annotation',
                'state_spec', 'state_protocol', 'checklist', 'annotation_path',
                'memory_annotation_path', 'state_spec_path'}


def bind_assets(value, root, key=''):
    if isinstance(value, dict):
        return {k: bind_assets(v, root, k) for k, v in value.items()}
    if isinstance(value, list):
        return [bind_assets(v, root, key) for v in value]
    if isinstance(value, str) and value and key in ASSET_FIELDS and not value.startswith('archive_ref:'):
        path = Path(value)
        return str(path if path.is_absolute() else root/path)
    return value


def load_dataset(path):
    root = Path(path).resolve()
    config = read(root / "dataset.json")
    catalog_path = Path(config["catalog"])
    records = read(catalog_path if catalog_path.is_absolute() else root / catalog_path)
    if not isinstance(records, list):
        raise ValueError("Dataset catalog must be an array")
    catalog = {r["case_id"]: bind_assets(r, root) for r in records}
    if len(catalog) != len(records):
        raise ValueError("Duplicate dataset case IDs")
    return root, config, catalog

def resolve(path, root):
    path = Path(path)
    return str(path if path.is_absolute() else root / path)


def validate_output_protocol(row, observation):
    """Check declared directory-input conventions during normal decoding."""
    protocol=row.get('output_protocol',{})
    reference=row['reference'];meta=observation['metadata']
    validate_automatic_protocol(protocol,meta)
    if protocol.get('match_reference_dimensions'):
        if (meta['width'],meta['height']) != (reference['width'],reference['height']):
            raise ValueError('Input binding error: video dimensions differ from the declared reference-native protocol; provide an explicit spatial mapping')
    if protocol.get('match_reference_timeline'):
        expected=(reference['frame_count']-1)/reference['fps']
        if meta['frame_count'] != reference['frame_count'] or abs(meta['times'][-1]-expected)>0.5/reference['fps']:
            raise ValueError('Input binding error: video timeline differs from the declared reference-native protocol; provide an explicit time mapping')

def normalize(args, *, dataset_info=None, cache_root=None):
    root, dataset, catalog = dataset_info or load_dataset(args.dataset)
    allowed = set(read(args.cases)) if args.cases else set(catalog)
    if allowed - catalog.keys():
        raise ValueError("Unknown cases: " + ", ".join(sorted(allowed - catalog.keys())))
    if args.predictions:
        value = read(args.predictions)
        rows = value["predictions"] if isinstance(value, dict) else value
    elif args.use_reference_videos:
        rows = [{"case_id": c, "video": catalog[c]["reference"]["video"], "video_sha256": catalog[c]["reference"].get("sha256"), "time_mapping": {"kind": "identity"}, "spatial_mapping": {"kind": "identity"}} for c in sorted(allowed)]
    else:
        folder = Path(args.videos).resolve()
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".mp4", ".mov", ".mkv", ".avi"))
        rows = []
        for file in files:
            cid = file.stem
            if cid not in catalog:
                raise ValueError("Unknown video case ID: " + cid)
            if cid not in allowed:
                continue
            sidecar = file.with_suffix(".json")
            row = read(sidecar) if sidecar.exists() else {}
            if row.get("case_id", cid) != cid:
                raise ValueError("Sidecar case ID mismatch: " + cid)
            row.update(case_id=cid, video=str(file))
            rows.append(row)
    normalized = []
    for value in rows:
        row = dict(value)
        cid = row["case_id"]
        if cid not in catalog:
            raise ValueError("Unknown case ID: " + cid)
        if cid not in allowed:
            continue
        if row.get("model", args.model) != args.model:
            raise ValueError("Each run must contain exactly one model")
        row.update(model=args.model, category=catalog[cid]["category"], role="prediction")
        row["video"] = resolve(row["video"], root)
        errors = [row['input_error']] if row.get('input_error') else []
        if not Path(row["video"]).is_file():
            errors.append("Video file is missing: " + row["video"])
        row.setdefault("reference", catalog[cid]["reference"])
        row['reference'] = bind_assets(row['reference'], root)
        if not row.get("video_sha256") and Path(row['video']).is_file():
            row["video_sha256"] = file_digest(row["video"])
        if not errors and (not row.get("time_mapping") or not row.get("spatial_mapping")):
            protocol = row.get("output_protocol", catalog[cid].get("standard_output_protocol", dataset.get("standard_output_protocol")))
            try:
                row = complete_mappings(row, protocol, cache_root=cache_root)
            except (ValueError, KeyError, TypeError) as exc:
                errors.append(str(exc))
        for field in ("time_mapping", "spatial_mapping"):
            if not row.get(field):
                errors.append("Missing " + field)
        if row.get('time_mapping'):
            from handworldbench.data.timeline import Timeline
            try: Timeline(row['time_mapping'])
            except (ValueError, KeyError, TypeError) as exc: errors.append(str(exc))
        if row.get('spatial_mapping',{}).get('kind')=='affine':
            from handworldbench.data.spatial_mapping import affine_matrix_3x3
            try:row['spatial_mapping']=dict(row['spatial_mapping'],matrix=affine_matrix_3x3(row['spatial_mapping'].get('matrix')).tolist())
            except (ValueError,TypeError) as exc:errors.append(str(exc))
        if row.get('spatial_mapping',{}).get('kind')=='opencv_calibrated':
            from handworldbench.observations.pinhole_calibration import validate_calibration
            try:validate_calibration(row['spatial_mapping']['spec'],source_size=[row['reference']['width'],row['reference']['height']])
            except (ValueError,KeyError,TypeError) as exc:errors.append(str(exc))
        if errors: row['input_error'] = '; '.join(dict.fromkeys(errors))
        row["metrics"] = select(row.get("metrics", args.metrics))
        normalized.append(row)
    if not normalized or len({r["case_id"] for r in normalized}) != len(normalized):
        raise ValueError("Require nonempty input with unique case IDs")
    return dataset, catalog, normalized

