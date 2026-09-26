"""Development-only fitting and isolated reference evaluation for intervention state."""
from pathlib import Path

import numpy as np
from scipy.signal import resample
from dataclasses import replace

from . import intervention_model as im
from . import present_state_model as pm
from . import geometry_model as gm
from .intervention_aware_state import ROOT, DEV, CFG, case, originals
from .geometry_evolution import development_sources, unforced
from .geometry_assay import initialize_written
from .present_state_transmission import old_initial, load, digest
from .hybrid_pair import save_npz_exclusive


def development(out,args,budget):
    inputs={}
    for row,path,index in development_sources():
        if row['seed'] not in DEV or row['history'] not in ('none','odd04') or row == {'seed':8101,'history':'odd04'}:
            continue
        inputs[str(path)]=digest(path)
        case(out,pm.load_checkpoint(path,50.,index),row,budget)
    pm.save_json(out/'inputs.json',inputs)


def records(data=None):
    result=[]
    for root in ([ROOT/'pilot-01',ROOT/'development-02'] if data is None else [data]):
        for p in sorted(root.glob('*gap*.npz')):
            prefix,gap=p.stem.split('-gap');s,h,a=prefix.split('-a')[0].split('-')[0],prefix.split('-a')[0].split('-')[1],prefix.split('-a')[1]
            with np.load(p,allow_pickle=False) as d:
                d=dict(d)
            d.update(seed=int(s[1:]),history=h,a=float(a),gap=int(gap),prefix=prefix,root=root)
            result.append(d)
    return result


def jumps(rows):
    result=[];seen=set()
    for r in rows:
        key=(r['seed'],r['history'],r['a'])
        if key not in seen:
            seen.add(key);result.append(r)
    return result


def fit_jump(rows,kind):
    r=jumps(rows)
    return im.fit_kick([x['z_before'] for x in r],[x['a'] for x in r],
                       [x['z_after']-x['z_before'] for x in r],kind)


def score(truth,pred,floors=None):
    floors=np.zeros((2,2)) if floors is None else np.asarray(floors)
    result=[]
    for k,(name,mask) in enumerate([('whole',pm.TIMES>=0),('late',pm.TIMES>=40)]):
        for label,t,p in [('R_without',truth[0],pred[0]),('R_after',truth[1],pred[1]),('D_event',truth[1]-truth[0],pred[1]-pred[0])]:
            for o,output in enumerate(['mass','moment']):
                actual=t[mask,o];estimate=p[mask,o];err=estimate-actual
                rms=float(np.sqrt(np.mean(actual**2)));error=float(np.sqrt(np.mean(err**2)))
                peak=int(np.argmax(abs(actual)));ph=pm.TIMES[mask]
                result.append(dict(kind=label,window=name,output=output,rms=rms,error_rms=error,
                    error_max=float(abs(err).max()),relative=error/rms if rms else None,
                    floor=float(floors[k,o]),resolved=bool(rms>floors[k,o]),
                    passed=bool(rms>floors[k,o] and error/rms <= (.1 if label=='D_event' else .02)),
                    true_peak_time=float(ph[peak]),predicted_peak_time=float(ph[np.argmax(abs(estimate))]),
                    true_peak_value=float(actual[peak]),predicted_at_true_peak=float(estimate[peak])))
    return result


def plain_fit(out,args,budget):
    budget.begin('grouped-jump-fits')
    rows=records();gs,F=originals();G=gs['quadratic-1e-06']
    scores=[];physical=[]
    for seed in DEV:
        train=[r for r in rows if r['seed']!=seed]
        test=[r for r in rows if r['seed']==seed]
        models={name:dict(G=G,F=F,J=None if name=='ignore' else fit_jump(train,name)) for name in ['ignore','constant','dependent']}
        for r in test:
            baseline=np.load(r['root']/f"s{r['seed']}-{r['history']}-a0-geometry.npz")['z'][r['gap']]
            for name,m in models.items():
                zs=[];ys=[]
                for a in [0.,r['a']]:
                    z,y=im.forecast(m,r['z0'],a,gaps=(r['gap'],));zs.append(z[0]);ys.append(y[0])
                for v in score(r['y'],np.array(ys)):
                    scores.append(dict(seed=seed,history=r['history'],a=r['a'],gap=r['gap'],model=name,**v))
                physical.append(dict(seed=seed,history=r['history'],a=r['a'],gap=r['gap'],model=name,
                    jump_error=(im.kick(m['J'],r['z_before'],r['a'])-r['z_after']).tolist(),
                    endpoint_error=(zs[1]-r['z_later']).tolist()))
            y=pm.predict(F,np.array([baseline,r['z_later']]))
            for v in score(r['y'],y):
                scores.append(dict(seed=seed,history=r['history'],a=r['a'],gap=r['gap'],model='exact_centers_F',**v))
            truejump=gm.rollout(G,r['z_after'],[float(r['gap'])])[0]
            physical.append(dict(seed=seed,history=r['history'],a=r['a'],gap=r['gap'],model='true_jump_G',endpoint_error=(truejump-r['z_later']).tolist()))
    pm.save_json(out/'scores.json',scores);pm.save_json(out/'physical.json',physical)
    pm.save_json(out/'models.json',{name:dict(G=G,F=F,J=None if name=='ignore' else fit_jump(rows,name)) for name in ['ignore','constant','dependent']})
    budget.finish()


def refine(out,args,budget):
    # Full write/wait/event/probe refinement, not a post-event reset.
    initial=old_initial(8101)
    save_npz_exclusive(out/'prepared-initial.npz',initial=initial)
    for label,config in [('halfdt',replace(CFG,dt=CFG.dt/2)),('doubleN',replace(CFG,n=CFG.n*2))]:
        target=out/label;target.mkdir()
        budget.begin(label+'-full-prefix')
        init=initial if config.n==CFG.n else resample(initial,config.n,axis=-1)
        state=unforced(initialize_written(init,'odd04',config),config,[0.,50.],budget)[0][-1]
        budget.finish()
        case(target,state,dict(seed=8101,history='odd04'),budget,config=config)
    base=ROOT/'pilot-01';floors=[];coords=[]
    for p in sorted(base.glob('*gap*.npz')):
        with np.load(p) as d:
            y=d['y'];absolute=d['absolute']
        for label in ['halfdt','doubleN']:
            with np.load(out/label/p.name) as d:
                err=d['y']-y
            floors.append([5*np.sqrt(np.mean(err[:,mask]**2,axis=1)).sum(axis=0) for mask in [pm.TIMES>=0,pm.TIMES>=40]])
        # Conservative arithmetic scale from the actual absolute measurement.
        floors.append(np.ones((2,2))*64*np.finfo(float).eps*abs(absolute[:,:,2,:2]).max(axis=(0,1)))
    for p in base.glob('*geometry.npz'):
        z=np.load(p)['z']
        for label in ['halfdt','doubleN']:
            coords.append(5*abs(np.load(out/label/p.name)['z']-z).max(axis=0))
    pm.save_json(out/'refinement.json',dict(response_floors=np.max(floors,axis=0).tolist(),coordinate_floors=np.max(coords,axis=0).tolist(),
        rule='max across matched refinements of 5*sum separate R RMS differences; minimum64*eps*max absolute C readout'))


def transient_fit(rows,order):
    """Small ERA realization from train-only physical impulse trajectories.

    No response error is used to fit the geometry. K(0) is constrained to the
    local measured-jump regression. The latent coordinates describe only the
    increment excited by a conditioning event, not complete field equilibration.
    """
    J=fit_jump(rows,'dependent');examples=jumps(rows)
    x=np.array([im.kick_features(r['z_before'],r['a'],J) for r in examples])
    curves=[]
    for r in examples:
        positive=np.load(r['root']/f"{r['prefix']}-geometry.npz")['z']
        baseline=np.load(r['root']/f"s{r['seed']}-{r['history']}-a0-geometry.npz")['z']
        curves.append(positive-baseline)
    curves=np.array(curves)
    K=np.linalg.solve(x.T@x+1e-6*np.eye(x.shape[1]),x.T@curves.reshape(len(x),-1)).reshape(x.shape[1],31,3).transpose(1,2,0)
    K[0]=0.;K[0,0]=J['coefficients']
    H0=np.block([[K[i+j] for j in range(10)] for i in range(10)])
    H1=np.block([[K[i+j+1] for j in range(10)] for i in range(10)])
    u,s,vt=np.linalg.svd(H0,full_matrices=False)
    u=u[:,:order];v=vt[:order];sq=np.sqrt(s[:order])
    A=(u.T@H1@v.T)/sq[:,None]/sq[None,:]
    C=u[:3]*sq
    # Equality-constrained fit: C B = physical K0, exactly.
    obs=np.concatenate([C@np.linalg.matrix_power(A,t) for t in range(31)])
    target=K.reshape(31*3,-1)
    block=np.block([[obs.T@obs,C.T],[C,np.zeros((3,3))]])
    B=np.linalg.solve(block,np.concatenate([obs.T@target,K[0]]))[:order]
    return dict(A=A.tolist(),B=B.tolist(),C=C.tolist(),J=J,order=order,
                spectral_radius=float(max(abs(np.linalg.eigvals(A)))),singular_values=s.tolist())


def corrected_response_fit(model,rows):
    features=[];targets=[]
    for r in rows:
        s0=im.State(model,r['z0']);s1=im.State(model,r['z0'])
        for s in [s0,s1]:s.advance(10)
        s0.event(0.);s1.event(r['a'])
        s0.advance(r['gap']);s1.advance(r['gap'])
        features.append(s1.memory)
        targets.append(r['y'][1]-r['y'][0]-(s1.response()-s0.response()))
    x=np.asarray(features);y=np.asarray(targets)
    scale_y=np.sqrt(np.mean(y*y,axis=(0,1)))
    _,s,vt=np.linalg.svd((y/scale_y).transpose(0,2,1).reshape(-1,161),full_matrices=False)
    basis=vt[:4]
    target=np.einsum('nto,kt->nok',y/scale_y,basis).reshape(len(y),-1)
    scale_x=np.maximum(np.sqrt(np.mean(x*x,axis=0)),1e-14)
    x=x/scale_x
    coef=np.linalg.solve(x.T@x+1e-6*np.eye(x.shape[1]),x.T@target)
    return dict(basis=basis.tolist(),coefficients=coef.tolist(),scale_x=scale_x.tolist(),scale_y=scale_y.tolist(),singular_values=s.tolist(),training_rows=len(y))


def repair_fit(out,args,budget):
    budget.begin('grouped-transient-and-readout-fits')
    rows=records();gs,F=originals();G=gs['quadratic-1e-06'];scores=[];physical=[];predictions=[]
    for seed in DEV:
        train=[r for r in rows if r['seed']!=seed];test=[r for r in rows if r['seed']==seed]
        for order in [4,6]:
            m=dict(G=G,F=F,J=fit_jump(train,'dependent'),transient=transient_fit(train,order))
            # Fit correction before adding it to model; unchanged F baseline stays exact.
            correction=corrected_response_fit(m,train)
            for corrected in [False,True]:
                if corrected:m['readout']=correction
                name=f"transient{order}"+('-readout' if corrected else '')
                for r in test:
                    zs=[];ys=[]
                    for a in [0.,r['a']]:
                        z,y=im.forecast(m,r['z0'],a,gaps=(r['gap'],));zs.append(z[0]);ys.append(y[0])
                    for v in score(r['y'],np.array(ys)):
                        scores.append(dict(seed=seed,history=r['history'],a=r['a'],gap=r['gap'],model=name,**v))
                    predictions.append(dict(seed=seed,history=r['history'],a=r['a'],gap=r['gap'],model=name,z=np.array(zs).tolist(),y=np.array(ys).tolist()))
                if not corrected:
                    for r in jumps(test):
                        state=im.State(m,r['z0']);state.advance(10);state.event(r['a'])
                        pred=[state.flow.z.copy()]
                        for _ in range(30):pred.append(state.advance(1))
                        actual=np.load(r['root']/f"{r['prefix']}-geometry.npz")['z']
                        err=np.array(pred)-actual
                        physical.append(dict(seed=seed,history=r['history'],a=r['a'],model=name,
                            max_error=abs(err).max(axis=0).tolist(),rms_error=np.sqrt(np.mean(err*err,axis=0)).tolist(),
                            spectral_radius=m['transient']['spectral_radius']))
    models={}
    for order in [4,6]:
        m=dict(G=G,F=F,J=fit_jump(rows,'dependent'),transient=transient_fit(rows,order))
        models[f'transient{order}']=dict(m)
        m['readout']=corrected_response_fit(m,rows)
        models[f'transient{order}-readout']=m
    pm.save_json(out/'models.json',models);pm.save_json(out/'scores.json',scores)
    pm.save_json(out/'physical.json',physical);pm.save_json(out/'predictions.json',predictions)
    budget.finish()
