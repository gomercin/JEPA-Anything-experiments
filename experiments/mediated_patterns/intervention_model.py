"""Compact event-aware runtime: descriptors/static coefficients only, no solver."""
import json

import numpy as np

from . import geometry_model as gm
from . import present_state_model as pm


def kick_features(z, amplitude, model):
    a = amplitude / .02
    x = np.r_[1., (np.asarray(z)-model['mean'])/model['scale']] if model['kind'] == 'dependent' else np.ones(1)
    return np.r_[a*x, a*a*x]


def fit_kick(z, amplitudes, jumps, kind):
    z, amplitudes, jumps = np.asarray(z),np.asarray(amplitudes),np.asarray(jumps)
    if kind not in ('constant','dependent') or z.shape != jumps.shape or z.shape[1:] != (3,):
        raise ValueError('Invalid local jump training rows')
    model = dict(kind=kind, mean=z.mean(axis=0).tolist(), scale=np.maximum(z.std(axis=0),1e-14).tolist())
    x = np.array([kick_features(v,a,model) for v,a in zip(z,amplitudes,strict=True)])
    model['coefficients'] = np.linalg.solve(x.T@x + 1e-6*np.eye(x.shape[1]), x.T@jumps[:,0]).tolist()
    return model


def kick(model, z, amplitude):
    if not np.isfinite(amplitude) or abs(amplitude) > .020000000001:
        raise ValueError('Conditioning amplitude outside qualified interval')
    z = np.asarray(z,float).copy()
    if z.shape != (3,) or not np.isfinite(z).all():
        raise ValueError('Three finite centers required')
    if model is not None:
        z[0] += kick_features(z,amplitude,model) @ np.asarray(model['coefficients'])
    return z


class State:
    """Original G with a local physical event map; no future schedule input."""
    def __init__(self, model, z):
        self.model = json.loads(json.dumps(model,allow_nan=False))
        self.flow = gm.Continuation(model['G'],z)

    def advance(self, steps):
        return self.flow.advance(steps)

    def event(self, amplitude):
        self.flow.z = kick(self.model['J'],self.flow.z,amplitude)

    def response(self):
        return pm.predict(self.model['F'],self.flow.z[None])[0]

    def checkpoint(self,path):
        pm.save_json(path,dict(schema=1,model_sha256=gm.identity(self.model),z=self.flow.z.tolist(),steps=self.flow.steps))

    @classmethod
    def restore(cls,model,checkpoint):
        if set(checkpoint) != {'schema','model_sha256','z','steps'} or checkpoint['schema'] != 1 or checkpoint['model_sha256'] != gm.identity(model):
            raise ValueError('Checkpoint/model mismatch')
        obj = cls(model,checkpoint['z'])
        if isinstance(checkpoint['steps'],bool) or not isinstance(checkpoint['steps'],int) or checkpoint['steps'] < 0:
            raise ValueError('Invalid counter')
        obj.flow.steps=checkpoint['steps']
        return obj


def forecast(model,z0,amplitude,delay=10,gaps=(10,30)):
    state = State(model,z0)
    state.advance(delay)
    state.event(amplitude)
    z, y = [],[]
    elapsed=0
    for gap in gaps:
        z.append(state.advance(gap-elapsed));y.append(state.response());elapsed=gap
    return np.array(z),np.array(y)
