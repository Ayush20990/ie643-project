"""Label isolation, geometry reconstruction and multivariate forecast preservation."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy.fft import idct
from catboost import CatBoostRegressor

from src.event_boost import EventBoostForecaster,features,input_pair,encode,decode,event_quality


class EventBoostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def cfg(self):return {"L":72,"H":24,"columns":["OT","load"],"key":"OT"}

    def test_features_exclude_withheld_suffix_and_other_windows(self):
        rng=np.random.default_rng(4);x=rng.normal(size=(3,72,8)).astype(np.float32)
        prefix,tail=input_pair(x,24,2)
        f=features(prefix,self.cfg())
        changed=x.copy();changed[0,-24:,:2]+=100;changed[1]+=100
        prefix2,tail2=input_pair(changed,24,2)
        g=features(prefix2,self.cfg())
        for a,b in zip(f,g):np.testing.assert_array_equal(a[0],b[0])
        self.assertFalse(np.array_equal(tail[0],tail2[0]))
        self.assertEqual(tail.shape,(3,24,2))
        self.assertTrue(all(np.isfinite(v).all() for v in f))

    def test_geometry_roundtrip_and_event_metric_penalizes_missed_peaks(self):
        rng=np.random.default_rng(9)
        coeff=np.zeros((4,24));coeff[:,:8]=rng.normal(size=(4,8))
        y=idct(coeff,norm="ortho",axis=1)
        offset=np.arange(4);scale=np.arange(1,5)
        encoded=encode(y,offset,scale,"geometry")
        p=decode(encoded,offset,scale,"geometry","mean")
        np.testing.assert_allclose(p,y,atol=2e-6)
        sharp=decode(encoded,offset,scale,"geometry","extrema")
        np.testing.assert_allclose(sharp,y,atol=2e-6)
        exact=event_quality(y,y)
        self.assertAlmostEqual(exact["score"],0.)
        low=y.copy();low[np.arange(4),y.argmax(1)]-=2
        self.assertGreater(event_quality(y,low)["event_MSE"],0.)

    def test_native_tree_roundtrip_and_only_temperature_changes(self):
        rng=np.random.default_rng(8)
        cfg=self.cfg();x=rng.normal(size=(6,72,8)).astype(np.float32)
        f,offset,scale=features(x,cfg)
        y=rng.normal(size=(6,24))
        tree=CatBoostRegressor(iterations=3,depth=2,border_count=8,loss_function="MultiRMSE",verbose=False,
                              allow_writing_files=False,thread_count=2)
        tree.fit(f,y)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"tree.cbm";tree.save_model(str(path))
            loaded=CatBoostRegressor();loaded.load_model(str(path))
            np.testing.assert_array_equal(tree.predict(f),loaded.predict(f))
        selection={"weight":.5,"variant":"trajectory","mode":"mean","trees":3}
        model=EventBoostForecaster(cfg,"residual",{},tree,selection).eval()
        t=torch.from_numpy(x)
        with torch.no_grad():base=model.base_prediction(t);p=model(t)
        torch.testing.assert_close(base[...,1],p[...,1],rtol=0,atol=0)
        expected=.5*base[...,0].numpy()+.5*decode(tree.predict(f),offset,scale,"trajectory")
        np.testing.assert_allclose(p[...,0],expected,atol=1e-6)
        model.selection["weight"]=0.
        torch.testing.assert_close(model(t),base,rtol=0,atol=0)


if __name__=="__main__":unittest.main()
