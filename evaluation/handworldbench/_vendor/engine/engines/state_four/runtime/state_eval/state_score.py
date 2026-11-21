# State scoring with immutable instance context.
import math,copy
from pathlib import Path
import cv2,numpy as np
from PIL import Image
from .io import ROOT,read,save,sha
from .sampling import sample_window
from .transitions import strata
from .retention_gate_data_driven import render_gate, validate_state_policies
from .score_normalization import publish
from .component_scores import aggregate,gated
from .progress_score import evaluate as evaluate_progress
from .legal_termination import PROTOCOL as LEGAL_TERMINATION_PROTOCOL, build_evidence as legal_termination_evidence, prompt_for as legal_termination_prompt
ANCHOR = '\nTracked entity, described neutrally before evaluation: {entity}\nTrack this SAME entity from the earliest image through the sequence. Its identity is not the hands, the whole scene, a collection of unrelated items, or whichever later object is easiest to explain. The description helps locate the reference entity; it does not assert that a requested state change occurred or that the current video matches the reference appearance. Track the actual initially visible entity. Appearance mismatch to the description alone is not a physical violation; judge temporal changes within this video. Ordinary rotation and occlusion are allowed. A sampling gap is not itself a violation, but is also not evidence that an unseen tear, assembly, replacement, insertion, or material transformation actually happened. Do not rescue an observed incompatible material or structural change by inventing such an operation or by redefining the tracked object. Judge only the visible change and its visible support. If essential identity or structure truly cannot be assessed, use the existing unknown code 0. An unchanged, identifiable object can still be physically consistent even with zero task progress.\n'
PREFIX = 'Judge only visible evidence. The requirement states what to evaluate, not what happened. Do not infer success from familiar hand activity or hidden geometry. '
END = 'Reply with a single digit only. Do not explain. 0 means the essential visual evidence cannot be assessed. Use the actual extent of fulfillment, not confidence, for grades 1 through 5.'
def immutable(path, value):
    if Path(path).exists():
        if read(path) != value:
            raise ValueError('Existing record differs: ' + str(path))
    else:
        save(path, value)
def grade_value(choice):
    if choice in ('1', '2', '3', '4', '5'):
        return (int(choice) - 1) / 4
    return None
def prefix_indices(fps, count, start):
    """Match the frozen four-uniform-prefix rule, including both endpoints."""
    if start <= 0:
        return []
    last = min(count - 1, math.ceil(start * fps - 1e-07) - 1)
    if fps <= 0 or last < 0:
        raise ValueError('Invalid source video interval')
    return sorted(set((int(round(x)) for x in np.linspace(0, last, min(4, last + 1)))))
def global_inputs(evidence):
    prefix = evidence['prefix']
    return (prefix + evidence['overview'], ['Prefix ' + str(i) for i in range(len(prefix))] + ['T' + str(i) for i in range(len(evidence['overview']))])

class StateEvaluator:
    def __init__(self,context,row,numeric_choose=None,binary_choose=None):
        self.context=context;self.row=row
        self.spec_path=context.field_path(row,'state_spec')
        self.protocol_path=context.field_path(row,'state_protocol')
        self.evidence_path=context.field_path(row,'state_evidence')
        self.out=context.output_root/'state'
        self.numeric_choose=numeric_choose;self.binary_choose=binary_choose
    def fingerprint(self,case):
        actual=sha(case['video'])
        if actual!=case['video_sha256']:raise ValueError('Video content differs')
        return {'case':case,'config_sha256':self.context.fingerprint,'specs_sha256':sha(self.spec_path),
                'protocol_sha256':sha(self.protocol_path),'roi_evidence_sha256':sha(self.evidence_path) if self.evidence_path.exists() else None,
                'model_binding':self.context.model_binding(),'actual_video_sha256':actual,
                'dependencies':{p.name:sha(p) for p in sorted((ROOT/'state_eval').glob('*.py'))}}
    def retention_only(self,judge,case,spec):
        validate_state_policies(spec)
        fp=self.fingerprint(case);final=self.out/case['case_id']/'summary.json'
        if final.exists():
            old=read(final)
            if old.get('fingerprint')!=fp:raise ValueError('Existing retention result differs')
            return old
        result={'fingerprint':fp,'case':case,'model':'Qwen3.5-9B','reason_generation':False,
                'metrics':{k:publish(None,'not_applicable',applicable=False) for k in ['state_progression','evolution_correctness']},
                'progress_support_used':False}
        if spec['applicability']['state_regulation']['status']!='applicable':
            result['metrics']['state_regulation']=publish(None,'not_applicable',applicable=False)
        else:
            evidence=read(self.evidence_path)
            if evidence['binding']['case']['video_sha256']!=case['video_sha256']:raise ValueError('ROI/video binding differs')
            windows={w['id']:w for w in evidence['windows']};frames={f['source_index']:f for f in evidence['frames']}
            protocol=read(self.protocol_path);ev={'retention':[]}
            for original in spec['persistence_units']:
                unit=copy.deepcopy(original);locator=protocol.get('target_locators',{}).get(case['clip_id'])
                if locator:unit['state_description']+=' Location and identity anchors, not evidence of successful detachment: '+locator
                prefix='persistence:'+unit['id']+':'
                def get(key):return [frames[i] for i in windows[prefix+key]['source_indices']]
                ev['retention'].append({'unit':unit,'before':get('before'),'established':get('established'),
                                        'checks':[{'check':c,'frames':get(c['id'])} for c in unit['checks']]})
            result['metrics']['state_regulation']=self.retention(judge,case,spec,ev,None)
            result['metrics']['state_regulation']['establishment_gate_source']='explicit_full_condition_neutral_labels_v1' if any(u.get('establishment',{}).get('level')=='full_condition' for u in spec['persistence_units']) else 'direct_initial_vs_established'
        immutable(final,result);return result

    def prepare(self, case, spec):
        path = self.context.cache_root / 'state_prepared' / (case['case_id'] + '.json')
        fp = self.fingerprint(case)
        if path.exists():
            d = read(path)
            if d.get('fingerprint') != fp:
                raise ValueError('Evidence binding differs')
            cached_frames = d['prefix'] + d['progress'] + d['overview'] + [f for xs in d['native'] for f in xs]
            for item in d['retention']:
                cached_frames += item['before'] + item['established'] + [f for c in item['checks'] for f in c['frames']]
            for f in cached_frames:
                if sha(f['path']) != f['sha256']:
                    raise ValueError('Cached evidence changed')
            return d
        if sha(case['video']) != case['video_sha256']:
            raise ValueError('Video changed')
        ep = self.evidence_path
        e = read(ep)
        if e['binding']['case']['video_sha256'] != case['video_sha256']:
            raise ValueError('ROI source differs')
        ws = {w['id']: w for w in e['windows']}
        fs = {f['source_index']: f for f in e['frames']}
        ids = [ws['state:before']['source_indices'][0]] + sorted(set(ws['state:process']['source_indices'] + ws['state:after']['source_indices']))
        progress = [fs[i] for i in ids]
        box = e['binding']['temporal_union_bbox']
        cap = cv2.VideoCapture(case['video'])
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = [spec['windows']['before'][0], spec['windows']['after'][1]]
        overview_ids, prefix_ids, native_ids = [], [], []
        if 'evolution_correctness' in self.context.metrics:
            overview_ids = sample_window(fps, count, interval, max_frames=12)
            prefix_ids = prefix_indices(fps, count, interval[0])
            first = math.ceil(interval[0] * fps - 1e-07)
            last = math.ceil(interval[1] * fps - 1e-07) - 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, first)
            prev = None
            diff = {}
            for idx in range(first, last + 1):
                ok, bgr = cap.read()
                if not ok:
                    raise ValueError('Decode failed')
                small = cv2.resize(cv2.cvtColor(bgr[box[1]:box[3], box[0]:box[2]], cv2.COLOR_BGR2GRAY), (128, 72)).astype(np.float32)
                if prev is not None:
                    diff[idx] = float(np.abs(small - prev).mean())
                prev = small
            native_ids = []
            for lo, hi in strata(first, last):
                peak = max(range(max(first + 1, lo), hi + 1), key=lambda i: diff[i])
                start = max(first, min(peak - 1, last - 2))
                native_ids.append([start, start + 1, start + 2])
        folder = self.context.cache_root / 'state_canonical_from_start/frames' / case['video_sha256'] / '_'.join(map(str, box))
        folder.mkdir(parents=True, exist_ok=True)
        frames = {}
        for idx in sorted(set(prefix_ids + overview_ids + [i for group in native_ids for i in group])):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, bgr = cap.read()
            if not ok:
                raise ValueError('Decode failed')
            h, w = bgr.shape[:2]
            if not 0 <= box[0] < box[2] <= w or not 0 <= box[1] < box[3] <= h:
                raise ValueError('ROI outside original image')
            p = folder / (str(idx) + '.png')
            im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).crop(box)
            if p.exists():
                if not np.array_equal(np.array(im), np.array(Image.open(p).convert('RGB'))):
                    raise ValueError('Existing prepared pixels differ')
            else:
                im.save(p)
            frames[idx] = {'path': str(p), 'sha256': sha(p), 'source_index': idx, 'time_s': idx / fps, 'bbox': box}
        cap.release()
        ret = []
        for unit in spec['persistence_units']:
            prefix = 'persistence:' + unit['id'] + ':'
            get = lambda name: [fs[i] for i in ws[prefix + name]['source_indices']]
            ret.append({'unit': unit, 'before': get('before'), 'established': get('established'), 'checks': [{'check': c, 'frames': get(c['id'])} for c in unit['checks']]})
        d = {'fingerprint': fp, 'case_id': case['case_id'], 'video_sha256': case['video_sha256'], 'protocol_sha256': sha(self.protocol_path), 'roi_evidence_sha256': sha(ep), 'specs_sha256': sha(self.spec_path), 'prefix': [frames[i] for i in prefix_ids], 'global_range': [0, spec['windows']['after'][1]], 'native_range': interval, 'progress': progress, 'overview': [frames[i] for i in overview_ids], 'native': [[frames[i] for i in ids] for ids in native_ids], 'retention': ret, 'source_fps': fps}
        immutable(path, d)
        return d

    def query(self, judge, case, key, prompt, frames, labels, choices=('0', '1', '2', '3', '4', '5'), *, return_record=False):
        from .numeric_answer import choose_budgeted
        choose = self.numeric_choose or choose_budgeted
        images = []
        for f, label in zip(frames, labels):
            if sha(f['path']) != f['sha256']:
                raise ValueError('Changed frame')
            images.append((label + '; source time ' + str(f['time_s']) + ' seconds', Image.open(f['path']).convert('RGB')))
        answer = choose(judge, prompt, images, {'case': case, 'key': key, 'frames': frames, 'protocol_sha256': sha(self.protocol_path), 'specs_sha256': sha(self.spec_path), 'runner_sha256': sha(__file__), 'run_fingerprint': self.fingerprint(case)}, self.out / case['case_id'] / (key + '.json'), labels=choices)
        if answer.get('status') != 'ok':
            raise RuntimeError(answer.get('error') or 'State judgment execution failed: '+str(answer.get('status')))
        return answer if return_record else answer['choice'] if answer['status'] == 'ok' else None

    def evaluate(self, judge, case, spec):
        final = self.out / case['case_id'] / 'summary.json'
        fp = self.fingerprint(case)
        if final.exists():
            d = read(final)
            if d.get('fingerprint') != fp:
                raise ValueError('Summary version/input differs')
            return d
        common = {'fingerprint': fp, 'case': case, 'protocol_sha256': sha(self.protocol_path), 'code_sha256': sha(__file__), 'model': 'Qwen3.5-9B', 'reason_generation': False, 'metrics': {}}
        if spec['applicability']['state_progression']['status'] != 'applicable':
            return self.retention_only(judge, case, spec)
        ev = self.prepare(case, spec)
        prog = evaluate_progress(judge, case, spec, self.out / 'progress_observations', evidence_path=self.evidence_path, specs_path=self.spec_path, output_root=self.context.output_root, choose_fn=self.binary_choose)
        parts = [{'id': x['id'], 'value': x['raw_score'], 'published_score': x['score'], 'original_status': x['original_status']} for x in prog['checkpoints']]
        common['metrics']['state_progression'] = prog
        if 'evolution_correctness' in self.context.metrics:
            physical = []
            for check in spec['correctness_checks'][:2]:
                if not isinstance(spec['neutral_entity'], str) or not spec['neutral_entity'].strip():
                    raise ValueError('Missing frozen neutral entity')
                prompt = PREFIX + ANCHOR.format(entity=spec['neutral_entity'].strip()) + 'These fixed chronological samples show the manipulated object and operation. Evaluate physical process, not task completion or visual attractiveness. A static unchanged object may be physically correct while making no progress. Sampling gaps alone do not prove a discontinuity.\nProperty: ' + check['description'] + '\nFailure: ' + check['fail'] + '\nPartial: ' + check['partial'] + '\nPass: ' + check['pass'] + '\nGrade 1: severe or pervasive clear violation; 2: substantial visible violations; 3: mixed physically consistent and inconsistent changes; 4: largely consistent with a minor localized violation; 5: the observed process supports this property without a visible violation. ' + END
                fs, labels = global_inputs(ev)
                answer = self.query(judge, case, 'physics_' + check['id'], prompt, fs, labels, return_record=True)
                v = answer['choice'] if answer['status'] == 'ok' else None
                value = grade_value(v)
                status = 'ok' if value is not None else 'insufficient_evidence' if answer['status'] == 'ok' else answer['status']
                physical.append({'id': check['id'], 'grade': v, 'value': value, 'published': publish(value, status)})
            native = []
            for k, fs in enumerate(ev['native']):
                prompt = PREFIX + 'A, B, C are consecutive native frames. Exact elapsed time between adjacent frames is ' + str(1 / ev['source_fps']) + ' seconds. Evaluate continuity of the manipulated object and state, including material creation, impossible interpenetration and unsupported state jumps. Do not invent unshown operations within this short interval. Ordinary rotation, motion blur and occlusion are not themselves physical violations.\nGrade 1: severe unsupported replacement or state jump; 2: substantial physically unsupported change; 3: localized clear violation in an otherwise continuous transition; 4: a minor localized physical violation; 5: continuous physically plausible transition. If essential geometry is invisible and cannot be assessed, use 0 rather than inventing a failure. ' + END
                v = self.query(judge, case, 'physics_native_' + str(k), prompt, fs, list('ABC'))
                native.append({'grade': v, 'value': grade_value(v)})
            local_items = [publish(x['value'], 'ok' if x['value'] is not None else 'insufficient_evidence') for x in native]
            local = aggregate(local_items)
            local['score'] = min((x['score'] for x in local_items))
            local['raw_score'] = None if local['zero_imputed'] else local['score']
            local['aggregation'] = 'minimum of predeclared local transitions'
            physical.append({'id': 'physical_process', 'value': local['score'], 'native_transitions': native, 'published': local})
            raw = aggregate([p['published'] for p in physical])
            common['metrics']['evolution_correctness'] = {**gated(prog, raw), 'dimensions': physical, **({'global_evaluation_range': ev['global_range'], 'native_evaluation_range': ev['native_range']} if ev['prefix'] else {})}
        if 'state_regulation' in self.context.metrics:
            common['metrics']['state_regulation'] = self.retention(judge, case, spec, ev, parts) if spec['applicability']['state_regulation']['status'] == 'applicable' else publish(None, 'not_applicable', applicable=False)
        immutable(final, common)
        return common

    def retention(self, judge, case, spec, ev, parts):
        units = []
        for item in ev['retention']:
            u = item['unit']
            fs = [item['before'][0]] + item['established']
            prompt = PREFIX + 'B is the initial state; E are later establishment images. Did the SAME object acquire the specified NEW state relative to B? Object presence alone, a rigid turn or pre-existing covered material is insufficient.\nNew state: ' + u['state_description'] + '\nReply with one digit only: 1=yes, an identifiable new part of the specified state is established; 2=no, the specified new state was not established; 0=essential evidence cannot determine establishment. Do not explain.'
            if parts is None:
                linked_id = None
                linked = None
                normalization = u.get('establishment')
                if normalization is not None and normalization.get('level') == 'full_condition':
                    prompt = render_gate(normalization, spec['checkpoints'], retention_only=True)
                    fs = sorted(fs, key=lambda f: (f['time_s'], f['source_index']))
                    gate = self.query(judge, case, 'retention_' + u['id'] + '_gate', prompt, fs, ['S' + str(i) for i in range(len(fs))], ('0', '1', '2'))
                else:
                    gate = self.query(judge, case, 'retention_' + u['id'] + '_gate', prompt, fs, ['B initial'] + ['E' + str(i) for i in range(len(item['established']))], ('0', '1', '2'))
            else:
                linked_id = read(self.protocol_path)['persistence_checkpoint_map'][u['property_id']]
                linked = next((x for x in parts if x['id'] == linked_id))
                if (linked['value'] if linked is not None else None) is None:
                    gate = None
                elif (linked['value'] if linked is not None else None) == 0:
                    gate = '2'
                else:
                    gate = self.query(judge, case, 'retention_' + u['id'] + '_gate', prompt, fs, ['B initial'] + ['E' + str(i) for i in range(len(item['established']))], ('0', '1', '2'))
            windows = []
            if gate == '1':
                for c in item['checks']:
                    fs = [item['before'][0]] + item['established'] + c['frames']
                    labels = ['B initial'] + ['E' + str(i) for i in range(len(item['established']))] + ['L' + str(i) for i in range(len(c['frames']))]
                    prompt = PREFIX + 'Compare B initial, E established, and L later. Score the remaining amount of the NEW configuration gained between B and E, not mere existence of the object or configuration already present in B. Track the same affected local region. Turning can reveal an already-open part without undoing a local join.\nTracked state: ' + u['state_description'] + '\nCheck: ' + c['check']['description'] + '\nPermitted reversals: ' + ' '.join(u['allowed_reversals']) + '\nGrade 1: complete unsupported reset to the initial state or loss/replacement; 2: only a small part of the new state remains; 3: a substantial part remains and a substantial part is lost; 4: most remains with a small unsupported loss; 5: the new state remains without unsupported loss. Code 6 is ONLY for a visibly demonstrated permitted reversal of the actual tracked state; this ends the keeping obligation and is exempt, not a high score. Hand motion, rotation or temporary hiding alone does not establish a reversal. An allowed action later cannot excuse an earlier unsupported reset of the same state. ' + END
                    v = self.query(judge, case, 'retention_' + u['id'] + '_' + c['check']['id'], prompt, fs, labels, ('0', '1', '2', '3', '4', '5', '6'))
                    if v == '6':
                        original = {'id': c['check']['id'], 'grade': v, 'exempt': True, 'value': None}
                        follow_frames, follow_labels, follow_sampling = legal_termination_evidence(self, case, u, c['check'], fs, labels)
                        follow = self.query(judge, case, 'retention_' + u['id'] + '_' + c['check']['id'] + '_legal_termination_v1', legal_termination_prompt(u, c['check']), follow_frames, follow_labels, ('0', '1', '2', '3', '4', '5'))
                        windows.append({'id': c['check']['id'], 'grade': follow, 'exempt': False, 'value': grade_value(follow), 'original_observation': original, 'repair_protocol': LEGAL_TERMINATION_PROTOCOL, 'repair_answer_status': 'ok', 'legal_termination_sampling': follow_sampling})
                    else:
                        windows.append({'id': c['check']['id'], 'grade': v, 'exempt': False, 'value': grade_value(v)})
            if gate == '2':
                score = publish(0.0, 'ok')
            elif gate != '1':
                score = publish(None, 'insufficient_evidence')
            else:
                score = aggregate([publish(w['value'], 'not_applicable' if w['exempt'] else 'ok' if w['value'] is not None else 'insufficient_evidence', applicable=not w['exempt']) for w in windows])
            units.append({'id': u['id'], 'linked_checkpoint_id': linked_id, 'linked_checkpoint_raw_score': linked['value'] if linked is not None else None, 'linked_checkpoint_score': linked['published_score'] if linked is not None else None, 'establishment': gate, 'windows': windows, **score})
        return {**aggregate(units), 'units': units}
