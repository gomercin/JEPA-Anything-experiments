"""Instrument invariants, not desired model superiority."""
import numpy as np
import pytest

from experiments.mediated_patterns import intervention_model as im
from experiments.mediated_patterns import present_state_model as pm
from experiments.mediated_patterns.intervention_aware_state import four_responses, jump_instrument
from experiments.mediated_patterns.source_receiver_relay import CFG


def specimen():
    x = np.arange(CFG.n)*CFG.length/CFG.n-CFG.length/2
    u=sum(np.exp(-((x-c)/4)**2)*np.cos(x-c+.1) for c in pm.ANCHORS)
    return np.array([u,.1*u*u])


@pytest.mark.parametrize('a',[0.,.02,-.02])
def test_exact_additive_center_identity_and_locality(a):
    original=specimen();after,info=jump_instrument(original,a)
    assert info['identity_max'] < 1e-13
    assert np.array_equal(after[1],original[1])
    assert np.array_equal(np.array(info['after'])[1:],np.array(info['before'])[1:])
    assert abs(info['l2']-abs(a)) < 1e-14


def test_four_branch_subtraction_removes_lingering_output():
    base=np.arange(12.).reshape(3,4);old=7*base;probe=2*base;interaction=.3*base
    y=np.stack([base,base+old,base+probe,base+old+probe+interaction],axis=1)
    r=four_responses(y)
    np.testing.assert_allclose(r[0],probe)
    np.testing.assert_allclose(r[1]-r[0],interaction)


def test_kicks_enforce_zero_and_locality_train_only_scaling():
    z=np.arange(18.).reshape(6,3)/20
    a=np.array([.02,-.02]*3);j=np.zeros_like(z);j[:,0]=a*z[:,0]
    model=im.fit_kick(z,a,j,'dependent')
    np.testing.assert_array_equal(model['mean'],z.mean(axis=0))
    for x in [z[0],np.array([50.,70.,90.])]:
        np.testing.assert_array_equal(im.kick(model,x,0),x)
        np.testing.assert_array_equal(im.kick(model,x,.02)[1:],x[1:])


def test_grouped_descendants_never_cross():
    groups=np.repeat([8101,10101,10102],12)
    for train,test in pm.grouped_folds(groups):
        assert not set(groups[train]) & set(groups[test])
