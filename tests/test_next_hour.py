"""Genuine one-hour output and observed-input self-supervision isolation."""
import inspect
import unittest

import torch

from src.nowcast_tcn import NextHourTCN,within_input,forecast_loss
from src.nowcast_train import adapt_fit


class NextHourTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def model(self):return NextHourTCN({"L":72,"columns":["temp","load"]})

    def test_initial_persistence_and_real_encoder_learning(self):
        torch.manual_seed(3);model=self.model().eval()
        x=torch.randn(4,72,8);y=torch.randn(4,1,2)
        torch.testing.assert_close(model(x),x[:,-1:,:2],rtol=0,atol=0)
        opt=torch.optim.AdamW(model.parameters(),lr=.001)
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            forecast_loss(model(x),y,x,0,True).backward()
            gradient=sum(float(p.grad.abs().sum()) for p in model.encoder.parameters() if p.grad is not None)
            opt.step()
        self.assertGreater(gradient,0.)
        self.assertEqual(model(x).shape,(4,1,2))

    def test_pretext_cannot_see_withheld_hour_or_other_windows(self):
        model=self.model().eval()
        with torch.no_grad():model.head[-1].weight.normal_(std=.01)
        x=torch.randn(3,72,8,requires_grad=True)
        prefix,tail=within_input(x,2);p=model(prefix)
        changed=x.detach().clone();changed[0,-1,:2]+=50;changed[1]-=50
        prefix2,_=within_input(changed,2)
        torch.testing.assert_close(p[0],model(prefix2)[0])
        p[0].sum().backward()
        self.assertEqual(x.grad[0,-1].abs().sum().item(),0.)
        self.assertEqual(x.grad[1].abs().sum().item(),0.)
        self.assertFalse(tail.requires_grad)
        names=inspect.signature(adapt_fit).parameters
        self.assertEqual([n for n in names if n.startswith('target')],['target_train_x','target_val_x'])


if __name__=="__main__":unittest.main()
