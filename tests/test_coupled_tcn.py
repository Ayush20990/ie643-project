"""Ablation and input isolation for the predicted-driver correction head."""
import unittest

import torch

from src.coupled_tcn import CoupledLiteTCN
from src.coupled_train import cache_input
from src.progressive_tcn import input_pair


class CoupledTCNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def model(self,control="coupled"):
        cfg={"L":48,"H":6,"columns":["load","OT"],"key":"OT"}
        return CoupledLiteTCN(cfg,"residual",control=control)

    def test_initial_exact_baseline_and_nonkey_outputs_unchanged_after_fit(self):
        model=self.model().eval()
        x=torch.randn(4,48,8)
        with torch.no_grad():
            torch.testing.assert_close(model(x),model.base(x[...,:2]))
        before={n:p.detach().clone() for n,p in model.base.named_parameters()}
        opt=torch.optim.AdamW(model.decoder.parameters(),lr=.01)
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            model(x).square().mean().backward();opt.step()
        with torch.no_grad():
            p=model(x);base=model.base(x[...,:2])
            torch.testing.assert_close(p[...,0],base[...,0],rtol=0,atol=0)
            self.assertGreater((p[...,1]-base[...,1]).abs().sum().item(),0.)
        self.assertTrue(all(torch.equal(p,before[n]) for n,p in model.base.named_parameters()))

    def test_control_removes_additional_driver_conditioning_and_tail_isolation(self):
        model=self.model("temperature_only").eval()
        with torch.no_grad():model.decoder[-1].weight.normal_()
        x=torch.randn(4,48,8)
        f,b,s=model.cache_features(x)
        changed=f.clone()
        for block in range(6):changed[...,block*2]+=100
        torch.testing.assert_close(model.from_features(f,b,s),model.from_features(changed,b,s))
        prefix,tail=input_pair(x,6)
        cache=cache_input(model,x)
        self.assertEqual(cache[-1].shape,(4,6,2))
        torch.testing.assert_close(cache[-1],tail[...,:2])
        altered=x.clone();altered[:,-6:,:2]+=100
        cache2=cache_input(model,altered)
        for first,second in zip(cache[:3],cache2[:3]):torch.testing.assert_close(first,second)


if __name__=="__main__":unittest.main()
