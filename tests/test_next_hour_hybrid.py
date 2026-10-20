"""Sparse neural/dynamics mixtures and local observed-history isolation."""
import unittest

import torch

from src.nowcast_hybrid import LocalDynamics,NextHourHybrid,dynamics_features
from src.nowcast_tcn import within_input
from src.horizon_reference import FirstHourReference
from src.model import make_forecaster


class NextHourHybridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def cfg(self):return {"L":72,"columns":["temp","load"]}

    def test_local_dynamics_forecast_units_and_tail_isolation(self):
        cfg=self.cfg();model=LocalDynamics(cfg)
        x=torch.randn(3,72,8,requires_grad=True)
        with torch.no_grad():model.weight.normal_(std=.03)
        prefix,_=within_input(x,2)
        f,last=dynamics_features(prefix,2)
        torch.testing.assert_close(model(prefix),last[:,None]+torch.einsum('bcf,cfh->bhc',f,model.weight))
        model(prefix)[0].sum().backward()
        self.assertEqual(x.grad[0,-1].abs().sum().item(),0.)
        self.assertEqual(x.grad[1:].abs().sum().item(),0.)

    def test_sparse_mixture_matches_components_and_roundtrips(self):
        cfg=self.cfg();model=NextHourHybrid(cfg).eval()
        with torch.no_grad():
            for name,child in model.models.items():
                if name.startswith('dynamics'):child.weight.normal_(std=.01)
                else:child.head[-1].weight.normal_(std=.01)
            model.mix.copy_(torch.tensor([[.1,.2,.3,.2,.2],[0.,.4,.3,.3,0.]]))
        x=torch.randn(4,72,8)
        with torch.no_grad():
            expected=(model.components(x)*model.mix[None,None]).sum(-1)
            torch.testing.assert_close(model(x),expected)
        loaded=NextHourHybrid(cfg).eval();loaded.load_state_dict(model.state_dict())
        torch.testing.assert_close(model(x),loaded(x))

    def test_reference_is_exactly_first_hour_of_existing_forecast(self):
        cfg={"L":72,"H":24,"columns":["temp","load"]}
        base=make_forecaster(cfg).eval()
        wrapper=FirstHourReference(base).eval()
        x=torch.randn(4,72,2)
        with torch.no_grad():torch.testing.assert_close(wrapper(x),base(x)[:,:1],rtol=0,atol=0)
        self.assertEqual(wrapper.H,1)


if __name__=='__main__':unittest.main()
