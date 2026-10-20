"""Causal event selection and source-only expert serialization."""
import unittest

import torch

from src.peak_nowcast import PeakNowcast,observed_regime,episode_prefix


class PeakNowcastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def model(self):return PeakNowcast({"L":72,"H":1,"columns":["temp","load"]},"context",{"architecture":"dlinear"}).eval()

    def test_episode_prefix_cannot_access_its_current_or_later_targets(self):
        model=self.model()
        x=torch.randn(3,72,8)
        prefix,truth=episode_prefix(x,55,2)
        changed=x.clone();changed[0,55:]+=100;changed[1]-=100
        other,other_truth=episode_prefix(changed,55,2)
        torch.testing.assert_close(prefix[0],other[0],rtol=0,atol=0)
        with torch.no_grad():torch.testing.assert_close(model(prefix)[0],model(other)[0],rtol=0,atol=0)
        self.assertFalse(torch.equal(truth[0],other_truth[0]))
        self.assertEqual(prefix.shape,(3,72,8))
        self.assertEqual(truth.shape,(3,1,2))

    def test_regime_weighting_and_clean_source_predictions(self):
        model=self.model()
        x=torch.randn(4,72,8)
        for i,last_three in enumerate(([1.,3.,4.],[4.,2.,1.],[1.,1.,1.],[1.,2.,4.])):
            x[i,-3:,:2]=torch.tensor(last_three)[:,None]
        regimes=observed_regime(x,2)
        self.assertEqual(regimes[:,0].tolist(),[1,2,3,0])
        with torch.no_grad():
            model.mix.uniform_(.1,1.)
            model.mix/=model.mix.sum(-1,keepdim=True)
            weights=model.mix[torch.arange(2)[None],regimes]
            expected=(model.experts(x)*weights[:,None]).sum(-1)
            torch.testing.assert_close(model(x),expected)
            model.mix[:,:,1]=0;model.mix[:,:,3]=0
            model.mix/=model.mix.sum(-1,keepdim=True)
            before=model(x)
            model.remove_inactive_target_weights()
            torch.testing.assert_close(model(x),before)


if __name__=="__main__":unittest.main()
