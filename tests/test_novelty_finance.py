import json, unittest
import numpy as np
import torch
from streamlit.testing.v1 import AppTest
from src.novelty_finance import windows,prepare_x,har_features,alignment,infer,DEST,ROOT

class NoveltyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.parts=windows('pharma')
        cls.bundle=json.loads((DEST/'seed_0.json').read_text())
    def test_chronology_and_series_boundaries(self):
        parts=self.parts
        self.assertLess(max(parts['train'][2]),min(parts['val'][2]))
        self.assertLess(max(parts['val'][2]),min(parts['test'][2]))
        self.assertEqual(len(parts['test'][0]),5*665)
        self.assertEqual(parts['test'][0].shape[1:],(60,4))
    def test_scale_equivariance(self):
        x=self.parts['test'][0][:8].copy(); xx=x.copy();xx[:,:,:3]*=2
        z,a=prepare_x(x);zz,aa=prepare_x(xx)
        np.testing.assert_allclose(z,zz,rtol=1e-5,atol=1e-6)
        np.testing.assert_allclose(aa,2*a,rtol=1e-6)
    def test_alignment_identity_and_gradients(self):
        s=torch.randn(20,16,requires_grad=True)
        for kind in ['mmd','coral']:
            self.assertAlmostEqual(alignment(s,s,kind).item(),0,places=5)
            loss=alignment(s,torch.randn(20,16),kind)
            self.assertTrue(torch.isfinite(loss));loss.backward(retain_graph=True)
        self.assertTrue(torch.isfinite(s.grad).all())
    def test_checkpoint_predictions_finite_positive(self):
        p=infer(self.parts['test'][0][:8],self.bundle)
        self.assertEqual(len(p),15)
        for n,a in p.items():
            self.assertEqual(a.shape,(8,));self.assertTrue(np.isfinite(a).all(),n);self.assertTrue((a>0).all(),n)
    def test_upload_future_excluded(self):
        x=self.parts['test'][0][0]; future=np.ones((2,4),dtype=np.float32)
        a=np.concatenate([x,future]);b=a.copy();b[60:]+=999
        pa=infer(a[:60][None],self.bundle);pb=infer(b[:60][None],self.bundle)
        for n in pa:np.testing.assert_array_equal(pa[n],pb[n])
    def test_novelty_page(self):
        app=AppTest.from_file(str(ROOT/'app/app.py'),default_timeout=90).run()
        app.checkbox(key='novelty_mode').check().run()
        self.assertEqual(len(app.exception),0)
        self.assertEqual(len(app.metric),4)
        [r for r in app.radio if r.label=='Finance forecast target'][0].set_value(5).run()
        self.assertEqual(len(app.exception),0)
        self.assertEqual(len(app.metric),4)
        [b for b in app.button if b.label=='Evaluate recent observed sessions'][0].click().run()
        self.assertEqual(len(app.exception),0)
        app.checkbox[1].check().run()
        self.assertEqual(len(app.exception),0)

if __name__=='__main__':unittest.main()
