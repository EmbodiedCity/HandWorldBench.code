"""Ordered criterion hierarchy. Full completion can only be tested after a minimum state is observed."""
def grade(minimum,full=None):
 if minimum not in ['1','2','3']:return {'status':'invalid_evaluation','score':None,'bounds':[0.,1.]}
 if minimum!='1' and full is not None:raise ValueError('Full criterion must not be queried after a failed or unknown minimum gate')
 if minimum=='2':return {'status':'ok','score':0.,'bounds':[0.,0.]}
 if minimum=='3':return {'status':'insufficient_evidence','score':None,'bounds':[0.,1.]}
 if full=='1':return {'status':'ok','score':1.,'bounds':[1.,1.]}
 if full=='2':return {'status':'ok','score':.5,'bounds':[.5,.5]}
 return {'status':'insufficient_evidence' if full=='3' else 'invalid_evaluation','score':None,'bounds':[.5,1.]}

def full_condition(checkpoint):
 return 'BOTH conditions must be visually demonstrated by the SAME new local state: (A) '+checkpoint['minimum_condition']+' (B) '+checkpoint['full_condition']
