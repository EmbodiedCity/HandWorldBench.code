"""Resident image-quality and video encoders."""
import collections,sys
from pathlib import Path
from handworldbench.backends.context import S,ENGINE,CONFIG,decoded_frames,stage_video

def run(task,models,out):
    row=task['row'];backend=task['backend'];engine=ENGINE
    if backend=='musiq':
        import os
        os.environ.update(TF_CPP_MIN_LOG_LEVEL='2',XLA_PYTHON_CLIENT_PREALLOCATE='false',JAX_PLATFORM_NAME='gpu')
        sys.path[:0]=[str(engine/'engines/classic/MUSIQ')]+CONFIG.get('musiq_python_paths',[])
        import numpy as np,jax,jax.numpy as jnp,tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        if not all(d.platform=='gpu' for d in jax.devices()):raise RuntimeError('MUSIQ GPU backend unavailable')
        if 'musiq' not in models:
            from musiq_image_eval import MusiqScorer
            checkpoint=Path(CONFIG['models'].get('musiq',S/'data/classic/MUSIQ/checkpoints/spaq_ckpt.npz'))
            scorer=MusiqScorer(engine/'engines/classic/MUSIQ',checkpoint,1)
            scorer.compiled_predict=jax.jit(lambda images:scorer.model.call(scorer.params,images));models['musiq']=scorer
        scorer=models['musiq'];frames=decoded_frames(row)['frames'];groups=collections.defaultdict(list);scores=[None]*len(frames)
        for i,f in enumerate(frames):
            im=scorer._official.prepare_image(str(f['path']),scorer.pp_config);groups[tuple(im.shape[1:])].append((i,im))
        batch=4
        for group in groups.values():
            offset=0
            while offset<len(group):
                block=group[offset:offset+batch]
                try:values=np.asarray(scorer.compiled_predict(jnp.concatenate([im for _,im in block],axis=0))).reshape(len(block),-1)[:,0]
                except Exception as exc:
                    if batch>1 and any(x in str(exc).lower() for x in ['memory','resource_exhausted']):batch=max(1,batch//2);continue
                    raise
                for (idx,_),score in zip(block,values):scores[idx]=float(score)
                offset+=len(block)
        native={'score':float(np.mean(scores)),'frame_scores':scores,'frame_count':len(frames),'batch':batch,'device':str(jax.devices())}
    elif backend=='pavrm':
        if 'pavrm' not in models:
            from handworldbench_aux.core import load_vendor
            from handworldbench_aux.automatic import NativeVideoProcessor
            from transformers import AutoProcessor,AutoModelForImageTextToText
            import torch
            module=load_vendor('WBench/src/metrics/physical/visual_plausibility.py','handworldbench_pavrm')
            path=str(CONFIG['models'].get('pavrm',S/'models/qwen3vl-a3b-visual-plausibility'))
            obj=object.__new__(module.PhysicalPlausibilityEvaluator);obj.device='cuda:0'
            obj.processor=NativeVideoProcessor(AutoProcessor.from_pretrained(path,local_files_only=True))
            obj.model=AutoModelForImageTextToText.from_pretrained(path,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':'cuda:0'}).eval()
            models['pavrm']=obj
        native=models['pavrm'].score_video(str(stage_video(row)),fps=2.)
        if native.get('error'):raise RuntimeError(native['error'])
    elif backend=='jepa':
        import numpy as np,torch,decord,jepa_metric as module
        from handworldbench.runtime.storage import cache_lock,digest,read,write
        decord.bridge.set_bridge('native')
        if 'jepa' not in models:models['jepa']=module.load_model(torch.device('cuda:0'))
        encoded={}
        for role,source in [('prediction',row),('reference',dict(row,video=row['reference']['video'],video_sha256=row['reference']['sha256']))]:
            model_binding=CONFIG['models'].get('jepa',str(Path(CONFIG['classic_data'])/'metric_suite/weights'))
            key=digest(dict(video=source['video_sha256'],role=role,protocol='worldarena16',models=model_binding))
            with cache_lock(Path(CONFIG['cache_root'])/'jepa_features',key) as folder:
                if (folder/'receipt.json').exists():encoded[role]=read(folder/'receipt.json');continue
                loader=module.WorldArenaVideo(stage_video(source));encoder,classifier=models['jepa'];features=[];samples=[]
                for index in range(len(loader)):
                    pixels,details=loader.get(index);features.append(module.encode(pixels,encoder,classifier,torch.device('cuda:0')));samples.append(details)
                if not features:raise ValueError('No native JEPA clips')
                array=np.concatenate(features,axis=0);dest=folder/'features.npy'
                if not np.isfinite(array).all():raise ValueError('Nonfinite JEPA features')
                np.save(dest,array,allow_pickle=False)
                encoded[role]=dict(feature_file=str(dest),shape=list(array.shape),sampling=samples,role=role,video_sha256=source['video_sha256'])
                write(folder/'receipt.json',encoded[role])
        native={'score':None,'status':'observed','measurement_level':'collection','features':encoded}

    return native
