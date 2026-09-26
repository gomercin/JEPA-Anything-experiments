"""Saved-data accounting only, no fitting or simulation."""
import json
import time
from pathlib import Path
import numpy as np
from experiments.mediated_patterns import present_state_model as m
root = Path('work/mediated_patterns/present_state_transmission')
out = root / 'accounting-02'
start = time.process_time()
load = lambda p: json.loads(p.read_text())
s = load(root / 'analysis-final/summary.json')
rows = load(root / 'fresh-01/rows.json')
model = load(root / 'frozen-01/models.json')['geometry']
state = m.load_checkpoint(root / 'fresh-01/s12103-history.npz', 100, 0)
loops = 1000
t = time.process_time()
for _ in range(loops): z = m.extract(state, feature_set='geometry')
extract_time = (time.process_time() - t) / loops
t = time.process_time()
for _ in range(loops): m.predict(model, z[None])
forecast_time = (time.process_time() - t) / loops
sco = s['scores']['geometry']
per_seed = {str(seed): {kind: {o: max(r['relative_rms'] for r in sco if r['seed']==seed and r['kind']==kind and r['output']==o) for o in ['mass','moment']} for kind in ['R','Delta_R']} for seed in [12101,12102,12103]}
per_window = {w: {kind: {o: max(r['relative_rms'] for r in sco if r['window']==w and r['kind']==kind and r['output']==o) for o in ['mass','moment']} for kind in ['R','Delta_R']} for w in ['whole','late']}
per_family = {f: {kind: {o: max(r['relative_rms'] for r in sco if r['history']==f and r['kind']==kind and r['output']==o) for o in ['mass','moment']} for kind in ['R','Delta_R']} for f in ['odd04','heldmix']}
# Projection diagnostic is evaluator-only, not an inference input.
fitroot=root/'balanced-fit-01'; fy=np.load(fitroot/'development.npz')['y']; fm=load(fitroot/'models.json')[load(fitroot/'fit.json')['best']['geometry']]
basis=np.array(fm['basis']); coeff=np.einsum('nto,kt->nok',fy/fm['scale_y'],basis); proj=np.einsum('nok,kt->nto',coeff,basis)*fm['scale_y']; fr=load(fitroot/'fit.json')['rows']; pairs=[]
for j,r in enumerate(fr):
 if r['history']!='none': pairs.append([next(i for i,a in enumerate(fr) if a['seed']==r['seed'] and a['wait']==r['wait'] and a['history']=='none'),j])
projection=m.scores(fy,proj,pairs,fr)
result={'per_seed':per_seed,'per_window':per_window,'per_family':per_family,
'geometry_extraction_cpu_per_call':extract_time,'geometry_forecast_cpu_per_call':forecast_time,'benchmark_repeats':loops,
'learned_numeric_scalars':sum(np.size(model[k]) for k in ['basis','coefficients','mean','scale','scale_y']),
'geometry_json_bytes':len(json.dumps(model,sort_keys=True,indent=2).encode()),
'full_current_state_float64_bytes':2*768*8,'forecast_values':161*2,'spatial_templates':0,'retained_examples':0,
'projection_worst':{k:max(r['relative_rms'] for r in projection if r['kind']==k) for k in ['R','Delta_R']},
'resolutions':{n:load(root/n/'refinement.json')['floors'] for n in ['dev-refine-01','fresh-refine-01']}}
m.save_json(out/'accounting.json',result)
m.save_json(out/'protocol.json',{'command':['accounting.py'],'scientific_execution':False,'analysis':'saved outcomes and fixed inference timings; no fit or solver'})
m.save_json(out/'budget.json',{'cpu_seconds':time.process_time()-start,'status':'COMPLETE'})
receipts={p.parent.name:load(p)['cpu_seconds'] for p in sorted(root.glob('*/budget.json'))}
m.save_json(root/'budget-total.json',{'stages':receipts,'metered_cpu_seconds':sum(receipts.values()),'inspection_and_startup_allowance_seconds':30,'charged_cpu_seconds':30+sum(receipts.values()),'limit_cpu_seconds':1800,'fresh_and_refinement_cpu_seconds':receipts['fresh-01']+receipts['fresh-refine-01']})
print(json.dumps(result,indent=2)); print(load(root/'budget-total.json'))
