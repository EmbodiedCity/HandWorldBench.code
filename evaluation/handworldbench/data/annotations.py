"""Metric-scoped annotation binding to a video's declared timeline."""
import copy
from pathlib import Path
from handworldbench.data.timeline import Timeline
from handworldbench.runtime.storage import read,write,digest

INTERVALS={'before','process','after','established','window','interval','task_interval','establishment_window','retention_window'}
TIMES={'start_s','end_s','start','end','start_seconds','end_seconds','time_s','timestamp_seconds'}

def document(value):
    return copy.deepcopy(value) if isinstance(value,dict) else read(value)

def convert(value,timeline,key=''):
    if isinstance(value,dict):return {k:convert(v,timeline,k) for k,v in value.items()}
    if isinstance(value,list):
        if key in INTERVALS and len(value)==2 and all(type(x) in (int,float) for x in value):return [timeline.at(x) for x in value]
        return [convert(v,timeline,key) for v in value]
    if type(value) in (int,float) and key in TIMES:return timeline.at(value)
    return copy.deepcopy(value)

def semantic(item,row,metric,folder):
    supplementary=item['annotations'].get('supplement',{}).get(metric,{}) if metric=='event_adherence' else {}
    original=document(supplementary['annotation']) if supplementary.get('annotation') else document(item['annotations']['semantic'])
    if not original:raise ValueError('Missing semantic annotation')
    timeline=Timeline(row['time_mapping']);result=convert(original,timeline)
    result['reference_sha256']=row['reference']['sha256']
    result['reference_duration_s']=timeline.at(original.get('reference_duration_s',item['reference']['frame_count']/item['reference']['fps']))
    path=Path(folder)/('annotation_'+digest(result)+'.json');write(path,result);return path

def memory(item,row,defaults,folder):
    metrics=('state_progression','evolution_correctness','state_regulation','memory_consistency')
    units=item['annotations'].get('supplement',{}).get('memory_consistency',{}).get('inputs')
    if units is None:units=[dict(annotation=v['memory_annotation'],memory_unit_checklists=v.get('memory_unit_checklists',{}),segmentation_concepts=v.get('segmentation_concepts',[])) for v in item['annotations']['views'] if v.get('memory_annotation') and 'memory_consistency' not in v.get('not_applicable',{})]
    if not units:raise ValueError('Applicable memory annotation has no units')
    config=copy.deepcopy(defaults);config.update(metrics=['memory_consistency'],cases=[],reuse_memory_cache=False)
    timeline=Timeline(row['time_mapping']);folder=Path(folder)
    for i,unit in enumerate(units):
        original=document(unit['annotation']);ann=convert(original,timeline);ann['reference_sha256']=row['reference']['sha256']
        ann['reference_duration_s']=timeline.at(original.get('reference_duration_s',item['reference']['frame_count']/item['reference']['fps']))
        path=folder/f'memory_{i}.json';write(path,ann);checklists=copy.deepcopy(unit.get('memory_unit_checklists',{}))
        case=dict(case_id=row['case_id']+f'__memory_{i}',clip_id=row['case_id'],parent_case_id=row['case_id'],model=row['model'],video=row['video'],video_sha256=row['video_sha256'],reference_video=row['reference']['video'],reference_sha256=row['reference']['sha256'],memory_annotation=str(path),memory_unit_checklists=checklists,segmentation_concepts=unit.get('segmentation_concepts',[]),robot_roi=row['category']!='human',time_mapping='identity',not_applicable={m:'Outside requested memory evidence' for m in metrics if m!='memory_consistency'})
        config['cases'].append(case)
    return config

def state(item,row,defaults,folder,metrics):
    """Bind each applicable state view independently to prediction time."""
    config=copy.deepcopy(defaults);config.update(metrics=list(metrics),cases=[])
    timeline=Timeline(row['time_mapping']);folder=Path(folder)
    for i,view in enumerate(item['annotations'].get('views',[])):
        if not view.get('state_spec') or all(m in view.get('not_applicable',{}) for m in metrics):continue
        spec=convert(document(view['state_spec']),timeline)
        spec.update(clip_id=row['case_id'],reference_sha256=row['reference']['sha256'],reference_duration_s=timeline.at(item['reference']['frame_count']/item['reference']['fps']))
        spec_path=folder/f'state_spec_{i}.json';write(spec_path,spec)
        protocol=document(view['state_protocol']);protocol_path=folder/f'state_protocol_{i}.json';write(protocol_path,protocol)
        case=dict(case_id=row['case_id']+f'__state_{i}',clip_id=row['case_id'],parent_case_id=row['case_id'],model=row['model'],video=row['video'],video_sha256=row['video_sha256'],reference_video=row['reference']['video'],reference_sha256=row['reference']['sha256'],state_spec=str(spec_path),state_protocol=str(protocol_path),segmentation_concepts=view.get('segmentation_concepts',[]),robot_roi=row['category']!='human',time_mapping='identity',not_applicable=copy.deepcopy(view.get('not_applicable',{})))
        for key in ('state_roi_normalization','task_family'):
            if key in view:case[key]=copy.deepcopy(view[key])
        config['cases'].append(case)
    if not config['cases']:raise ValueError('Applicable state metric has no bound annotation view')
    return config

def content_signature(value):
    """Bind small annotation content once, without hashing model weights."""
    if isinstance(value,dict):return {k:content_signature(v) for k,v in value.items()}
    if isinstance(value,list):return [content_signature(v) for v in value]
    if isinstance(value,str) and value.endswith('.json') and Path(value).is_file():
        return dict(path=value,sha256=digest(read(value)))
    return value
