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


def corrected_response_fit(model,rows,interaction=False,event_basis=False):
    features=[];targets=[];centers=[]
    for r in rows:
        s0=im.State(model,r['z0']);s1=im.State(model,r['z0'])
        for s in [s0,s1]:s.advance(10)
        s0.event(0.);s1.event(r['a'])
        s0.advance(r['gap']);s1.advance(r['gap'])
        a=r['a']/.02;d=a*np.exp(-r['gap']/20.)
        features.append(np.array([a,d,a*a,a*d]) if event_basis else s1.memory);centers.append(s1.flow.z.copy())
        targets.append(r['y'][1]-r['y'][0]-(s1.response()-s0.response()))
    x=np.asarray(features);y=np.asarray(targets)
    scale_y=np.sqrt(np.mean(y*y,axis=(0,1)))
    _,s,vt=np.linalg.svd((y/scale_y).transpose(0,2,1).reshape(-1,161),full_matrices=False)
    basis=vt[:4]
    target=np.einsum('nto,kt->nok',y/scale_y,basis).reshape(len(y),-1)
    scale_x=np.ones(x.shape[1]) if event_basis else np.maximum(np.sqrt(np.mean(x*x,axis=0)),1e-14)
    x=x/scale_x
    centers=np.array(centers);mean_z=centers.mean(axis=0);scale_z=np.maximum(centers.std(axis=0),1e-14)
    if interaction:
        q=(centers-mean_z)/scale_z
        x=np.einsum('ni,nj->nij',x,np.c_[np.ones(len(x)),q]).reshape(len(x),-1)
    coef=np.linalg.solve(x.T@x+1e-4*np.eye(x.shape[1]),x.T@target)
    return dict(event_basis=event_basis,interaction=interaction,mean_z=mean_z.tolist(),scale_z=scale_z.tolist(),basis=basis.tolist(),coefficients=coef.tolist(),scale_x=scale_x.tolist(),scale_y=scale_y.tolist(),singular_values=s.tolist(),training_rows=len(y))


def repair_fit(out,args,budget):
    budget.begin('grouped-transient-and-readout-fits')
    rows=records();gs,F=originals();G=gs['quadratic-1e-06'];scores=[];physical=[];predictions=[]
    for seed in DEV:
        train=[r for r in rows if r['seed']!=seed];test=[r for r in rows if r['seed']==seed]
        for order in [4,6]:
            m=dict(G=G,F=F,J=fit_jump(train,'dependent'),transient=transient_fit(train,order))
            # Fit correction before adding it to model; unchanged F baseline stays exact.
            correction=corrected_response_fit(m,train,interaction=True,event_basis=True)
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
        m['readout']=corrected_response_fit(m,rows,interaction=True,event_basis=True)
        models[f'transient{order}-readout']=m
    pm.save_json(out/'models.json',models);pm.save_json(out/'scores.json',scores)
    pm.save_json(out/'physical.json',physical);pm.save_json(out/'predictions.json',predictions)
    budget.finish()


def freeze(out,args,budget):
    from .intervention_aware_state import FRESH, F_FILE, G_FILE
    models=load(ROOT/'plain-fit-01/models.json')
    models.update(load(args.data/'models.json'))
    gs,F=originals();models['affine-ignore']=dict(G=gs['affine-1e-06'],F=F,J=None)
    pm.save_json(out/'models.json',models)
    refinement=load(ROOT/'refine-01/refinement.json')
    pm.save_json(out/'freeze.json',dict(selected='transient6-readout',models_sha256=digest(out/'models.json'),
        original_G_path=str(G_FILE),original_G_sha256=digest(G_FILE),original_F_path=str(F_FILE),original_F_sha256=digest(F_FILE),
        sources={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},
        fresh_seeds=FRESH,development_seeds=DEV,histories=['none','odd04'],t0=50,delay=10,gaps=[10,30],
        amplitudes=[.02,-.02,.01],withheld='intermediate conditioning amplitude +.01; no timing change',
        response_floors=refinement['response_floors'],coordinate_floors=refinement['coordinate_floors'],
        coordinate_tolerances=[.00005,.00002,.000002],R_limit=.02,D_limit=.1,
        no_fresh_repair=True,refinement_rule=refinement['rule']))


def fresh(out,args,budget):
    import time
    from .intervention_aware_state import FRESH
    from .organization_response import isolated_components
    from .source_receiver_relay import prepare
    from .simulator import Field
    contract=load(args.freeze/'freeze.json');models=load(args.freeze/'models.json')
    if digest(args.freeze/'models.json')!=contract['models_sha256'] or contract['sources']!={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}:
        raise ValueError('Frozen code/models changed')
    if contract['fresh_seeds']!=FRESH or set(FRESH)&set(DEV):raise ValueError('Split mismatch')
    rows=[];starts=[]
    for seed in FRESH:
        f=Field(CFG);initial=prepare(f,isolated_components(f,seed,budget),seed,-4.,budget)
        save_npz_exclusive(out/f's{seed}-initial.npz',initial=initial)
        for history in contract['histories']:
            budget.begin(f'prepare-{seed}-{history}')
            state=unforced(initialize_written(initial,history,CFG),CFG,[0.,50.],budget)[0][-1]
            rows.append(dict(seed=seed,history=history));starts.append(state);budget.finish()
    save_npz_exclusive(out/'boundary-50.npz',states=starts)
    z0=np.array([pm.extract(s,feature_set='geometry') for s in starts])
    save_npz_exclusive(out/'initial-descriptors.npz',z=z0)
    pm.save_json(out/'rows.json',rows)
    budget.begin('seal-all-future-predictions')
    start=time.process_time();cpu=[]
    for i,z in enumerate(z0):
        for name,model in models.items():
            zs=[];ys=[];trajectories=[]
            for a in [0.,*contract['amplitudes']]:
                s=im.State(model,z);s.advance(contract['delay']);s.event(a)
                traj=[s.flow.z.copy()];pred=[]
                for n in range(1,31):
                    s.advance(1);traj.append(s.flow.z.copy())
                    if n==2:s.checkpoint(out/f'checkpoint-{i}-{name}-a{a:g}.json')
                    if n in contract['gaps']:pred.append(s.response())
                trajectories.append(traj);zs.append(np.array(traj)[contract['gaps']]);ys.append(pred)
            save_npz_exclusive(out/f'prediction-{i}-{name}.npz',z=zs,y=ys,trajectory=trajectories)
    cpu.append(time.process_time()-start)
    pm.save_json(out/'seal.json',dict(predictions={p.name:digest(p) for p in sorted(out.glob('prediction-*.npz'))},
        initialization_sha256=digest(out/'initial-descriptors.npz'),models_sha256=digest(args.freeze/'models.json'),
        checkpoint_hashes={p.name:digest(p) for p in sorted(out.glob('checkpoint-*.json'))},
        inference_cpu_seconds=sum(cpu),order='ALL reduced forecasts saved before generating ANY post-t0 field arrays or responses'))
    budget.finish()
    for state,row in zip(starts,rows,strict=True):
        case(out,state,row,budget,delay=contract['delay'],gaps=tuple(contract['gaps']),amplitudes=tuple(contract['amplitudes']))


def analyze(out,args,budget):
    budget.begin('scores-and-figures')
    contract=load(args.freeze/'freeze.json');data=args.data;models=load(args.freeze/'models.json')
    rows=records(data);initial_rows=load(data/'rows.json');all_scores=[];physical=[]
    for r in rows:
        i=initial_rows.index(dict(seed=r['seed'],history=r['history']));ai=[0.,*contract['amplitudes']].index(r['a']);gi=contract['gaps'].index(r['gap'])
        noevent=np.load(data/f"s{r['seed']}-{r['history']}-a0-geometry.npz")['z'][r['gap']]
        for name in [*models,'exact_centers_F']:
            if name=='exact_centers_F':
                _,F=originals();pred=pm.predict(F,np.array([noevent,r['z_later']]))
            else:
                p=np.load(data/f'prediction-{i}-{name}.npz');pred=p['y'][[0,ai],gi]
                physical.append(dict(seed=r['seed'],history=r['history'],a=r['a'],gap=r['gap'],model=name,
                    error=(p['z'][ai,gi]-r['z_later']).tolist(),true_event_motion=(r['z_later']-noevent).tolist(),
                    predicted_event_motion=(p['z'][ai,gi]-p['z'][0,gi]).tolist()))
            for s in score(r['y'],pred,contract['response_floors']):
                all_scores.append(dict(seed=r['seed'],history=r['history'],a=r['a'],gap=r['gap'],model=name,**s))
    pm.save_json(out/'scores.json',all_scores);pm.save_json(out/'physical.json',physical)
    summary={name:{kind:{o:max(s['relative'] for s in all_scores if s['model']==name and s['kind']==kind and s['output']==o) for o in ['mass','moment']} for kind in ['R_without','R_after','D_event']} for name in [*models,'exact_centers_F']}
    pm.save_json(out/'summary.json',summary)
    # Static scientific figures: each preparation remains a separate curve.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,6),layout='constrained')
    for r in rows:
        if r['history']!='odd04' or r['a']!=.02:continue
        i=initial_rows.index(dict(seed=r['seed'],history=r['history']));gi=contract['gaps'].index(r['gap'])
        p=np.load(data/f"prediction-{i}-{contract['selected']}.npz")['y'];pred=p[[0,1],gi];actual=r['y']
        for o in range(2):
            color='C'+str(r['seed']-16101)
            axes[0,o].plot(pm.TIMES,actual[1,:,o],color=color,alpha=.7)
            axes[0,o].plot(pm.TIMES,pred[1,:,o],color=color,ls='--')
            axes[1,o].plot(pm.TIMES,actual[1,:,o]-actual[0,:,o],color=color,alpha=.7,label=f"{r['seed']} gap{r['gap']}")
            axes[1,o].plot(pm.TIMES,pred[1,:,o]-pred[0,:,o],color=color,ls='--')
    for o,label in enumerate(['C mass','C signed moment']):
        axes[0,o].set_title(label+' R2 after');axes[1,o].set_title(label+' D_event');axes[1,o].set_xlabel('h after diagnostic probe')
    axes[1,0].legend(fontsize=7,ncol=2);fig.suptitle('True (solid) and predicted (dashed); +.02 conditioning, old odd write')
    fig.savefig(out/'responses.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,6),layout='constrained')
    for i,row in enumerate(initial_rows):
        if row['history']!='odd04':continue
        name=f"s{row['seed']}-{row['history']}"
        actual=np.load(data/f'{name}-a0.02-geometry.npz')['z'];base=np.load(data/f'{name}-a0-geometry.npz')['z']
        pred=np.load(data/f"prediction-{i}-{contract['selected']}.npz")['trajectory'];color='C'+str(row['seed']-16101)
        for o in range(3):
            axes[0,o].plot(np.arange(31),actual[:,o],color=color);axes[0,o].plot(np.arange(31),pred[1,:,o],color=color,ls='--')
            axes[1,o].plot(np.arange(31),actual[:,o]-base[:,o],color=color);axes[1,o].plot(np.arange(31),pred[1,:,o]-pred[0,:,o],color=color,ls='--')
            axes[0,o].set_title('ABC'[o]+' offset');axes[1,o].set_title('ABC'[o]+' event-induced motion');axes[1,o].set_xlabel('time after conditioning')
    fig.suptitle('Post-event geometry: true solid, predicted dashed; +.02, old odd write')
    fig.savefig(out/'geometry.png',dpi=160);plt.close(fig)
    budget.finish()
