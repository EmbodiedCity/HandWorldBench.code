"""Generic evidence preparation: explicit timing, SAM3 localization, expanded crops.

Memory windows use reference seconds and declared frame mapping. State windows
must use generated-video seconds; identity-in-seconds mappings are verified,
otherwise the configuration must explicitly declare already mapped windows.
"""
import math
from pathlib import Path
import cv2,numpy as np
from PIL import Image
from .io import read,save,sha
from .sampling import sample_window,task_windows
from .localization import Localizer,fallback_box,select,selected_hands
from .hand_bridge import selection_binding,require_current_evidence

def union(boxes):return [min(x[0] for x in boxes),min(x[1] for x in boxes),max(x[2] for x in boxes),max(x[3] for x in boxes)]
def probe(path):
    cap=cv2.VideoCapture(str(path));fps=float(cap.get(cv2.CAP_PROP_FPS));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));size=[int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))];cap.release()
    if not math.isfinite(fps) or fps<=0 or count<1 or min(size)<1:raise ValueError('Invalid video metadata')
    return fps,count,size

def frame_mapping(context,row,fps,count):
    if row.get('frame_mapping'):
        path=context.field_path(row,'frame_mapping');record=read(path)
        indices=record.get('source_frame_indices',record.get('reference_frame_indices'))
        if not isinstance(indices,list) or len(indices)!=count:raise ValueError('Explicit frame mapping length mismatch')
        if row.get('reference_video'):ref_fps,ref_count,_=probe(context.field_path(row,'reference_video'))
        else:ref_fps=float(record['reference_fps']);ref_count=int(record['reference_frame_count'])
        provenance={'kind':'explicit_indices','source':str(path),'source_sha256':sha(path)}
    elif row.get('time_mapping')=='identity':
        indices=list(range(count));ref_fps=fps;ref_count=count;provenance={'kind':'explicit_identity_declaration'}
    else:raise ValueError('Provide frame_mapping or explicitly declare time_mapping=identity')
    if not math.isfinite(ref_fps) or ref_fps<=0:raise ValueError('Invalid reference fps')
    if any(type(x) is not int or x<0 or x>=ref_count for x in indices):raise ValueError('Invalid source frame index')
    if any(b<a for a,b in zip(indices,indices[1:])):raise ValueError('Nonmonotonic source frame mapping')
    return np.array(indices),ref_fps,provenance

def object_box(candidates,size):
    hands=selected_hands(candidates)
    objects=[c for c in candidates if c['concept']!='hand']
    if not objects:
        b,r=fallback_box(hands,size);return b,{'object_found':False,'fallback':r}
    center=np.mean([[(c['bbox'][0]+c['bbox'][2])/2,(c['bbox'][1]+c['bbox'][3])/2] for c in hands],axis=0) if hands else np.array(size)/2
    def distance(c):
        b=c['bbox'];return np.linalg.norm(np.array([(b[0]+b[2])/2,(b[1]+b[3])/2])-center)
    obj=min(objects,key=lambda c:(distance(c),-c['confidence']));b=obj['bbox'];mx=max(32,.25*(b[2]-b[0]));my=max(32,.25*(b[3]-b[1]))
    return [max(0,math.floor(b[0]-mx)),max(0,math.floor(b[1]-my)),min(size[0],math.ceil(b[2]+mx)),min(size[1],math.ceil(b[3]+my))],{'object_found':True,'chosen':obj,'margin_fraction_each_side':.25,'min_margin_px':32}

def image_at(cap,index):
    cap.set(cv2.CAP_PROP_POS_FRAMES,int(index));ok,bgr=cap.read()
    if not ok:raise ValueError('Cannot decode frame '+str(index))
    return Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))

def localized_frames(cap,indices,localizer,concepts,crop=None):
    """Collect bounded batches within one pixel coordinate system and stage."""
    indices=list(indices);batch=max(1,int(localizer.batch_size))
    for start in range(0,len(indices),batch):
        block=indices[start:start+batch]
        images=[image_at(cap,index) for index in block]
        if crop is not None:images=[image.crop(crop) for image in images]
        candidates=localizer.candidates_many(images,concepts)
        if len(candidates)!=len(block):raise ValueError('Evidence localization output count mismatch')
        yield from zip(block,images,candidates)

def write_image(path,image):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        with Image.open(path) as old:
            if not np.array_equal(np.asarray(old.convert('RGB')),np.asarray(image)):raise ValueError('Existing evidence pixels differ')
    else:image.save(path)

def safe_destination(context,path):
    path=path.resolve()
    if not path.is_relative_to(context.output_root.resolve()) and not path.is_relative_to(context.cache_root.resolve()):
        raise ValueError('New evidence metadata must remain in configured output/cache roots')

def prepare_memory(context,row,localizer):
    case=context.case(row);path=context.field_path(row,'memory_evidence')
    if path.exists():
        e=read(path)
        require_current_evidence(e,context)
        if e['video_sha256']!=case['video_sha256']:raise ValueError('Existing memory evidence belongs to another video')
        from .pipeline import Pipeline
        Pipeline(context)._memory_binding(row,context.memory_annotation(row),e)
        if row.get('memory_segmentation_concepts',row.get('segmentation_concepts')) and e.get('concepts') and row.get('memory_segmentation_concepts',row.get('segmentation_concepts'))!=e['concepts']:raise ValueError('Existing memory concepts differ')
        return e
    safe_destination(context,path);annotation=context.memory_annotation(row);fps,count,size=probe(case['video'])
    indices,ref_fps,mapping=frame_mapping(context,row,fps,count);times=indices/ref_fps
    concepts=row.get('memory_segmentation_concepts',row.get('segmentation_concepts',[]))
    if not concepts:raise ValueError('Explicit segmentation concepts are required; no semantic auto-labeling')
    hidden={};needed=set()
    for unit in annotation['memory']:
        ids=unit.get('hidden_window_ids',row.get('memory_hidden_windows',{}).get(unit['id'],[]))
        if not ids:raise ValueError('Explicit hidden window IDs required for each memory unit')
        hidden[unit['id']]=ids;needed.update(ids+unit['history_window_ids']+unit['reappearance']['window_ids'])
        for q in unit['questions']:needed.update(q['window_ids'])
    windows={w['id']:w for w in annotation['windows']}
    if not needed<=set(windows):raise ValueError('Unknown memory window ID')
    cap=cv2.VideoCapture(case['video']);frames={};out_windows={};folder=context.cache_root/'memory_frames'/case['case_id']
    try:
        window_indices={};unique_indices={}
        for wid in sorted(needed):
            w=windows[wid];eligible=np.flatnonzero((times>=w['start_s']-1e-8)&(times<w['end_s']-1e-8));limit=2 if wid in sum(hidden.values(),[]) else 3
            picked=eligible[np.unique(np.linspace(0,len(eligible)-1,min(limit,len(eligible))).round().astype(int))] if len(eligible) else []
            window_indices[wid]=[int(index) for index in picked]
            for ix in window_indices[wid]:unique_indices.setdefault(ix,None)
        for ix,im,candidates in localized_frames(cap,unique_indices,localizer,concepts):
            box,detail=object_box(candidates,im.size)
            full=folder/(str(ix)+'_full.png');write_image(full,im)
            frames[ix]={'full_path':str(full),'full_sha256':sha(full),'frame_index':ix,'source_frame_index':int(indices[ix]),'source_time':float(times[ix]),'generated_time':ix/fps,'image_size':size,'individual_bbox':box,'candidates':candidates,'localization':detail}
        out_windows={wid:[frames[ix] for ix in picked] for wid,picked in window_indices.items()}
        if not frames:raise ValueError('No memory evidence frames')
        boxes=[f['individual_bbox'] for f in frames.values()]
        boxes += [c['bbox'] for f in frames.values() for c in selected_hands(f['candidates'])]
        b=union(boxes);mx=max(32,.15*(b[2]-b[0]));my=max(32,.15*(b[3]-b[1]));common=[max(0,int(b[0]-mx)),max(0,int(b[1]-my)),min(size[0],math.ceil(b[2]+mx)),min(size[1],math.ceil(b[3]+my))]
        for ix,f in frames.items():
            crop=folder/(str(ix)+'_roi.png')
            with Image.open(f['full_path']) as full:write_image(crop,full.crop(common))
            f.update(path=str(crop),sha256=sha(crop),bbox=common)
        out={'case_id':case['case_id'],'clip_id':case['clip_id'],'video_sha256':case['video_sha256'],
             'annotation_sha256':sha(context.field_path(row,'memory_annotation')),'preparer_sha256':sha(__file__),
             'concepts':concepts,'localizer':localizer.binding,'mapping':mapping,'reference_fps':ref_fps,'generated_fps':fps,
             'windows':out_windows,'unit_hidden_windows':hidden,'common_bbox':common,'crop_protocol':'temporal_union_hand_context',
             'status':'ready','hand_selection_binding':selection_binding(context)}
        save(path,out);return out
    finally:cap.release()

def retained_target_box(candidates,size,normalization):
    """Keep released targets even when hands move away; ambiguous/missing target keeps full context."""
    if normalization.get('version')!='released-target-union-v1' or not normalization.get('target_concepts'):
        raise ValueError('Unknown or incomplete state ROI normalization')
    targets=[c for c in candidates if c['concept'] in normalization['target_concepts']]
    if not targets:return [0,0,*size],{'target_found':False,'fallback':'full_frame_target_not_detected'}
    b=union([c['bbox'] for c in targets]);mx=max(48,.25*(b[2]-b[0]));my=max(48,.25*(b[3]-b[1]))
    box=[max(0,math.floor(b[0]-mx)),max(0,math.floor(b[1]-my)),min(size[0],math.ceil(b[2]+mx)),min(size[1],math.ceil(b[3]+my))]
    return box,{'target_found':True,'targets':targets,'margin_fraction':.25,'minimum_margin_px':48}

def prepare_state(context,row,localizer):
    case=context.case(row);path=context.field_path(row,'state_evidence')
    if path.exists():
        e=read(path)
        require_current_evidence(e,context)
        if e['binding']['case']['video_sha256']!=case['video_sha256']:raise ValueError('Existing state evidence belongs to another video')
        if e['binding'].get('specs_sha256') and e['binding']['specs_sha256']!=sha(context.field_path(row,'state_spec')):raise ValueError('Existing state specifications differ')
        if e['binding'].get('state_roi_normalization')!=row.get('state_roi_normalization'):raise ValueError('Existing state ROI normalization differs')
        if row.get('segmentation_concepts') and e['binding'].get('concepts') and row['segmentation_concepts']!=e['binding']['concepts']:raise ValueError('Existing state concepts differ')
        return e
    safe_destination(context,path);spec=context.state_spec(row);fps,count,size=probe(case['video']);indices,ref_fps,mapping=frame_mapping(context,row,fps,count)
    if not np.allclose(indices/ref_fps,np.arange(count)/fps,atol=1e-6,rtol=0) and row.get('state_windows_time_domain')!='generated':
        raise ValueError('State windows require explicitly mapped generated seconds; duration-ratio alignment is forbidden')
    concepts=row.get('segmentation_concepts',spec.get('concepts',[]))
    if not concepts:raise ValueError('Explicit segmentation concepts required')
    windows=task_windows(spec);requests={}
    for w in windows:
        w['source_indices']=sample_window(fps,count,w['window_s'],3)
        for ix in w['source_indices']:requests.setdefault(ix,[]).append(w['id'])
    if not requests:raise ValueError('No applicable state windows')
    cap=cv2.VideoCapture(case['video'])
    try:
        context_indices=set()
        for w in windows:
            start,end=w['window_s'];lo=math.ceil(start*fps-1e-6);hi=min(count,math.ceil(end*fps-1e-6))
            xs=sorted(set([lo,hi-1]+[min(hi-1,max(lo,round(t*fps))) for t in np.arange(start,end,1/3)]))
            context_indices.update([xs[round(i*(len(xs)-1)/5)] for i in range(6)] if len(xs)>6 else xs)
        contexts=[]
        for ix,im,all_candidates in localized_frames(cap,sorted(context_indices),localizer,concepts):
            hands=selected_hands(all_candidates)
            if hands:
                b=union([h['bbox'] for h in hands]);mx=max(24,.2*(b[2]-b[0]));my=max(24,.2*(b[3]-b[1]));box=[max(0,math.floor(b[0]-mx)),max(0,math.floor(b[1]-my)),min(size[0],math.ceil(b[2]+mx)),min(size[1],math.ceil(b[3]+my))]
            else:box=[0,0,*size]
            target_detail={}
            if row.get('state_roi_normalization'):
                target_box,target_detail=retained_target_box(all_candidates,size,row['state_roi_normalization']);box=union([box,target_box]);target_detail['required_bbox']=target_box
            contexts.append({'source_index':ix,'bbox':box,'hands':hands,'retained_target':target_detail})
        base=union([x['bbox'] for x in contexts]);details=[]
        for ix,crop,candidates in localized_frames(cap,sorted(requests),localizer,concepts,crop=base):
            uses=requests[ix];picked=select(candidates,crop.size);b=picked['bbox']
            details.append({'source_index':ix,'time_s':ix/fps,'uses':uses,**picked,'framewise_bbox_in_source':[b[0]+base[0],b[1]+base[1],b[2]+base[0],b[3]+base[1]],'candidates':candidates})
        common=union([f['framewise_bbox_in_source'] for f in details])
        if row.get('state_roi_normalization'):common=union([common]+[c['retained_target']['required_bbox'] for c in contexts])
        folder=context.cache_root/('state_frames_roi_released_target_v1' if row.get('state_roi_normalization') else 'state_frames_roi')/case['case_id']
        for f in details:
            im=image_at(cap,f['source_index']).crop(common);p=folder/(str(f['source_index'])+'_roi.png');write_image(p,im)
            f.update(path=str(p),sha256=sha(p),bbox_in_source=common,source_size=size,detail_size=list(im.size))
        result={'binding':{'case':case,'specs_sha256':sha(context.field_path(row,'state_spec')),'preparer_sha256':sha(__file__),
                           'state_roi_normalization':row.get('state_roi_normalization'),'localizer':localizer.binding,'concepts':concepts,'mapping':mapping,'temporal_union_bbox':common,'context_bbox_pixels':base},
                'windows':windows,'frames':details,'context_localization':contexts,'status':'prepared','not_a_score':True,'hand_selection_binding':selection_binding(context)}
        save(path,result);return result
    finally:cap.release()

def prepare(context,localizer_factory=Localizer):
    from .pipeline import Pipeline,PENDING
    pipeline=Pipeline(context);localizer=None;records=[]
    try:
        for row in context.config['cases']:
            case=context.case(row)
            if case['status'] in PENDING:continue
            if sha(case['video'])!=case['video_sha256']:raise ValueError('Video bytes changed')
            app=pipeline.applicability(row)
            for family,metrics,function in [('memory',['memory_consistency'],prepare_memory),('state',['state_progression','evolution_correctness','state_regulation'],prepare_state)]:
                if not any(app.get(m,{}).get('status')=='applicable' for m in metrics):continue
                key=family+'_evidence'
                if not context.field_path(row,key).exists() and localizer is None:localizer=localizer_factory(context)
                finder=getattr(localizer,'hand_finder',None)
                before=dict(getattr(finder,'stats',{}))
                evidence=function(context,row,localizer)
                if family=='memory':
                    from .memory_score import prepare_bundle
                    prepare_bundle(context,row,evidence)
                stats={k:v-before.get(k,0) for k,v in getattr(finder,'stats',{}).items()}
                records.append({'case_id':case['case_id'],'family':family,'evidence':str(context.field_path(row,key)),
                                'hand_detection':stats})
        return records
    finally:
        if localizer is not None and hasattr(localizer,'close'):localizer.close()
