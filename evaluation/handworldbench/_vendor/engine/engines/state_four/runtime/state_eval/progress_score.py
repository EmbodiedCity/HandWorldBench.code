"""State progression from annotated checkpoints and image evidence."""
from pathlib import Path
from PIL import Image
from .io import ROOT, read, sha
from .prompts import prompt
from .hierarchy import grade, full_condition
from .score_normalization import publish
from .component_scores import aggregate

VERSION = 'state-progress-checkpoints-1'
RUBRIC = 'minimum visible change is a necessary gate; full condition includes that same minimum and completion requirement'
LABELS = ('1', '2', '3')


def _destination(run_dir, output_root):
    directory = Path(run_dir).resolve()
    results = Path(output_root).resolve()
    if directory == results or not directory.is_relative_to(results):
        raise ValueError('Choose an explicit candidate subdirectory of results')
    return directory


def evaluate(judge, case, spec, run_dir, *, choose_fn=None,
             evidence_path=None, specs_path=None, output_root=None):
    """Return a published checkpoint aggregate without changing existing runs.

    With choose_fn omitted this calls classify.choose (no thinking, one digit).
    Injection accepts the same signature and must return status and choice.
    Unknown/invalid checkpoints receive zero individually; valid peers survive.
    Files and case video are hash-checked even for an injected replay.
    """
    directory = _destination(run_dir, output_root)
    live = choose_fn is None
    if live:
        from .binary_answer import choose
        choose_fn = choose
    if specs_path is None or evidence_path is None:raise ValueError('Explicit specs and evidence paths required')
    specs_path = Path(specs_path)
    common = {
        'metric': 'state_progression', 'version': VERSION, 'case': case,
        'protocol': 'minimum/full checkpoint grading',
        'execution_kind': 'classify_choose_or_identical_binding_resume' if live else 'injected_test_answers',
        'reason_generation': False,
        'component_code_sha256': sha(__file__),
        'specs_sha256': sha(specs_path),
        'hierarchy_sha256': sha(ROOT / 'state_eval/hierarchy.py'),
        'prompt_source_sha256': sha(ROOT / 'state_eval/prompts.py'),
        'classifier_sha256': sha(ROOT / 'state_eval/binary_answer.py'),
        'aggregation_sha256': sha(ROOT / 'state_eval/component_scores.py'),
        'publication_sha256': sha(ROOT / 'state_eval/score_normalization.py'),
    }
    if spec['applicability']['state_progression']['status'] != 'applicable':
        return {**common, **publish(None, 'not_applicable', applicable=False),
                'status': 'not_applicable', 'checkpoints': [], 'actual_source_frames': []}
    evidence_path = Path(evidence_path)
    evidence = read(evidence_path)
    if evidence['binding']['case']['video_sha256'] != case['video_sha256'] or sha(case['video']) != case['video_sha256']:
        raise ValueError('Video/evidence mismatch')
    windows = {w['id']: w for w in evidence['windows']}
    lookup = {f['source_index']: f for f in evidence['frames']}
    from .progress_frames import progress_indices
    ids = progress_indices(windows)
    frames = [lookup[i] for i in ids]
    if not spec['checkpoints']:
        raise ValueError('Applicable progress requires checkpoints')
    images = []
    try:
        for k, frame in enumerate(frames):
            if sha(frame['path']) != frame['sha256']:
                raise ValueError('Changed image')
            with Image.open(frame['path']) as source:
                rgb = source.convert('RGB')
            images.append(('Initial photograph' if k == 0 else 'Later observation ' + str(k), rgb))
        records = []
        for checkpoint in spec['checkpoints']:
            answers = {}
            minimum = full = None
            for level in ['minimum', 'full']:
                if level == 'full' and minimum != '1':
                    break
                condition = checkpoint['minimum_condition'] if level == 'minimum' else full_condition(checkpoint)
                binding = {
                    'case': case, 'specs_sha256': common['specs_sha256'],
                    'evidence_sha256': sha(evidence_path), 'frames': frames,
                    'checkpoint': checkpoint, 'level': level, 'mode': 'real',
                    'code_sha256': common['component_code_sha256'],
                    'hierarchy_sha256': common['hierarchy_sha256'],
                    'prompt_source_sha256': common['prompt_source_sha256'],
                    'rubric': RUBRIC, 'component_version': VERSION,
                    'execution_kind': common['execution_kind'],
                }
                dest = directory / case['case_id'] / 'real' / (checkpoint['id'] + '__' + level + '.json')
                # Default label tuple matches the frozen choose call exactly.
                answer = choose_fn(judge, prompt(spec['neutral_entity'], condition), images, binding, dest)
                if answer.get('status') != 'ok':
                    raise RuntimeError(answer.get('error') or 'Checkpoint judgment execution failed: '+str(answer.get('status')))
                choice = answer.get('choice') if answer.get('status') == 'ok' else None
                answers[level] = {
                    'choice': choice, 'raw_choice': answer.get('choice'),
                    'status': answer.get('status', 'invalid_evaluation'),
                    'path': str(dest), 'sha256': sha(dest) if live and dest.exists() else None,
                }
                if level == 'minimum':
                    minimum = choice
                else:
                    full = choice
            raw = grade(minimum, full)
            published = publish(raw['score'], raw['status'])
            records.append({
                'id': checkpoint['id'], 'answers': answers,
                'raw_judgment': raw, **published, 'status': published['original_status'],
            })
        total = aggregate(records)
        return {
            **common, **total, 'status': total['original_status'], 'checkpoints': records,
            'evidence_record': str(evidence_path), 'evidence_record_sha256': sha(evidence_path),
            'actual_source_frames': ids,
            'actual_source_times_s': [f['time_s'] for f in frames],
            'image_labels': [label for label, _ in images],
        }
    finally:
        for _, image in images:
            image.close()
