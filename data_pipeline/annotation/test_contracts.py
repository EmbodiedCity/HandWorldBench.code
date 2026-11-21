import unittest,tempfile,json,threading
from pathlib import Path
from types import SimpleNamespace
from annotate import Engine
from caption_contract import CODES,check_text
class Response:
 def __init__(self,v):self.v=v
 def raise_for_status(self):pass
 def json(self):return {'choices':[{'finish_reason':'stop','message':{'content':json.dumps(self.v)}}],'usage':{}}
class Session:
 def __init__(self,v):self.v=v;self.payload=None
 def post(self,url,**kwargs):self.payload=kwargs['json'];return Response(self.v)
class Contracts(unittest.TestCase):
 def engine(self,v):
  e=Engine.__new__(Engine);e.a=SimpleNamespace(model='test-model');e.endpoint='http://localhost';e.codes=CODES;e.lock=threading.Lock();e.stats=dict(requests=0,http_attempts=0,prompt_tokens=0,completion_tokens=0,retries=0);e.fake=Session(v);e.session=lambda:e.fake;return e
 def test_fresh_classification_and_same_images(self):
  e=self.engine(dict(primary='M1',evidence_frames=[0,80],rationale='Visible grasp and movement.'));frames=[0,20,40,60,80];images={k:'image'+str(k) for k in frames}
  with tempfile.TemporaryDirectory() as d:
   v=e.classify(Path(d),frames,images,'Use visible evidence.',10,'Generated global prose.',[dict(subtask_text='Visible local context.',clip_frames=[0,81])])
   self.assertEqual(v['primary'],'M1');self.assertEqual(v['sampled_clip_frames'],frames)
  q=e.fake.payload;self.assertTrue(q['chat_template_kwargs']['enable_thinking']);self.assertEqual(q['max_tokens'],4096);self.assertEqual([x['image_url']['url'] for x in q['messages'][1]['content'] if x['type']=='image_url'],[images[k] for k in frames])
 def test_bad_cached_code_rejected(self):
  e=self.engine({})
  with tempfile.TemporaryDirectory() as d:
   (Path(d)/'operation.json').write_text(json.dumps(dict(primary='INVALID',evidence_frames=[0],rationale='x',sampled_clip_frames=[0])))
   with self.assertRaises(ValueError):e.classify(Path(d),[0],{0:'x'},'',0,'x',[dict(subtask_text='x',clip_frames=[0,81])])
 def test_unshown_cached_evidence_rejected(self):
  e=self.engine({})
  with tempfile.TemporaryDirectory() as d:
   (Path(d)/'operation.json').write_text(json.dumps(dict(primary='M1',evidence_frames=[100],rationale='x',sampled_clip_frames=[0])))
   with self.assertRaises(ValueError):e.classify(Path(d),[0],{0:'x'},'',0,'x',[dict(subtask_text='x',clip_frames=[0,81])])
 def test_taxonomy_has_23_options(self):self.assertEqual(len(CODES),23)
 def test_two_paragraph_local_prose(self):
  check_text('Visible desk and object.\n\nThe hand moves the object.',False)
  with self.assertRaises(ValueError):check_text('Only one paragraph.',False)
class FullMockFlow(unittest.TestCase):
 def test_cpu_video_caption_and_new_classification_without_prior_labels(self):
  import subprocess,hashlib,shutil
  from caption_contract import check_caption
  ffmpeg=shutil.which('ffmpeg')
  if not ffmpeg:self.skipTest('FFmpeg is required')
  class Fake:
   def get(self,url,**kwargs):
    value={'data':[{'id':'test-model'}]} if url.endswith('/v1/models') else {'version':'test-runtime'}
    return SimpleNamespace(raise_for_status=lambda:None,json=lambda:value)
   def post(self,url,**kwargs):
    q=kwargs['json'];text=q['messages'][1]['content'][0]['text']
    if 'Choose one dominant visible operation' in text:return Response(dict(primary='M1',evidence_frames=[0],rationale='Visible test evidence.'))
    prose=' '.join(['visible']*85) if 'ENTIRE video' in text else 'Visible environment and objects.\n\nVisible local manipulation and result.'
    return SimpleNamespace(raise_for_status=lambda:None,json=lambda:{'choices':[{'finish_reason':'stop','message':{'content':prose}}],'usage':{}})
  old=Engine.session;Engine.session=lambda self:Fake()
  try:
   with tempfile.TemporaryDirectory() as td:
    r=Path(td);video=r/'test.mp4'
    subprocess.run([ffmpeg,'-v','error','-threads','1','-f','lavfi','-i','color=c=blue:s=64x64:r=30','-frames:v','81','-c:v','mpeg4',str(video)],check=True)
    (r/'service.json').write_text(json.dumps(dict(served_model='test-model',server_version={'version':'test-runtime'})))
    args=SimpleNamespace(out=r/'out',taxonomy=Path(__file__).with_name('taxonomy.json'),service_manifest=r/'service.json',endpoint='http://127.0.0.1:1',model='test-model',max_side=720,jpeg_quality=90,thinking=True,max_tokens=8192,concurrency=4)
    row=dict(id='test_clip',domain='ego',frames=81,source_frames=[0,81],video=str(video),video_sha=hashlib.sha256(video.read_bytes()).hexdigest())
    e=Engine(args)
    try:result=e.clip(row)
    finally:e.pool.shutdown(wait=True)
    c=json.loads(Path(result['caption']).read_text());check_caption(c,row)
    self.assertEqual(c['operation']['code'],'M1');self.assertEqual(len(c['segments']),3);self.assertEqual(e.stats['requests'],5)
    self.assertEqual(c['operation_provenance']['mode'],'new_visual_classification')
    args.preserve_operation=True
    row['previous_operation']=dict(code='M2',evidence_clip_frames=[2],rationale='Earlier visible evidence.')
    row['previous_operation_provenance']=dict(source_manifest_sha256='a'*64,model='prior')
    e=Engine(args)
    try:retained=e.clip(row)
    finally:e.pool.shutdown(wait=True)
    preserved=json.loads(Path(retained['caption']).read_text())
    check_caption(preserved,row)
    self.assertEqual(e.stats['requests'],4)
    self.assertEqual(preserved['operation'],row['previous_operation'])
    self.assertEqual(preserved['operation_provenance'],row['previous_operation_provenance'])
    self.assertNotEqual(retained['signature'],result['signature'])
    self.assertFalse(Path(retained['caption']).with_name('operation.json').exists())
  finally:Engine.session=old
if __name__=='__main__':unittest.main()
