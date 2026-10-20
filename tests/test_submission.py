"""Review-facing app checks, causal evaluation and exported result alignment."""
import io
import unittest

import numpy as np
import pandas as pd
import torch
from streamlit.testing.v1 import AppTest

from src.config import ROOT, SAMPLE_DIR
from src.submission import (evaluate_history, load_selected, predict_window,
                            read_input, score_forecasts)


class SubmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.meta,cls.models=load_selected("climate",1,"Peak-calibrated hybrid")
        cls.frame=read_input(SAMPLE_DIR/"climate_domain2_delhi.csv","climate")

    def test_forecast_excludes_future_rows(self):
        changed=self.frame.copy();changed.iloc[168:]+=999
        a=predict_window(self.frame,self.meta,self.models,"target",168)[3]
        b=predict_window(changed,self.meta,self.models,"target",168)[3]
        for key in a:np.testing.assert_array_equal(a[key],b[key])

    def test_export_matches_direct_predictions_and_truth(self):
        scores,export,origins=evaluate_history(self.frame.iloc[:176],self.meta,self.models,"target",maximum=3)
        np.testing.assert_array_equal(origins,[173,174,175])
        self.assertEqual(len(export),3*7)
        scaler,_,_,pred=predict_window(self.frame,self.meta,self.models,"target",173)
        np.testing.assert_allclose(export.adapted_model.iloc[:7],scaler.inverse(pred['adapted_model'])[0,0],rtol=1e-6)
        np.testing.assert_allclose(export.actual.iloc[:7],self.frame.iloc[173],rtol=1e-6)
        self.assertTrue(np.isfinite(scores.to_numpy()).all())

    def test_csv_rejects_duplicate_headers_and_hourly_gaps(self):
        with self.assertRaisesRegex(ValueError,"duplicate"):
            read_input(io.BytesIO(b'date,temperature_2m,temperature_2m\n2024-01-01,1,2\n'),"climate")
        broken=self.frame.iloc[:170].drop(self.frame.index[100])
        with self.assertRaisesRegex(ValueError,"consecutive"):
            read_input(io.BytesIO(broken.to_csv(index_label='date').encode()),"climate")

    def test_no_actuals_and_horizon_mismatch(self):
        scaler,x,y,p=predict_window(self.frame.iloc[:168],self.meta,self.models,"target",168)
        self.assertIsNone(y)
        self.assertTrue(score_forecasts(x,y,p,self.meta,scaler).empty)
        with self.assertRaisesRegex(ValueError,"horizons"):
            score_forecasts(x,np.zeros((1,24,7)),p,self.meta,scaler)

    def test_assessment_app_domain_and_horizon_combinations(self):
        app=AppTest.from_file(str(ROOT/"app"/"app.py"),default_timeout=90).run()
        checked=0
        for ds in ("climate","energy","finance"):
            app.selectbox(key="dataset").set_value(ds).run()
            for h in ([1,24] if ds!="finance" else [5]):
                app.radio(key=f"horizon_{ds}").set_value(h).run()
                for family in list(app.selectbox(key=f"model_{ds}_{h}").options):
                    app.selectbox(key=f"model_{ds}_{h}").set_value(family).run()
                    for role in ("Domain 1 · source","Domain 2 · target"):
                        app.radio(key="input_domain").set_value(role).run()
                        self.assertEqual(list(app.exception),[],(ds,h,family,role))
                        self.assertEqual(len(app.metric),4)
                        checked+=1
        print(f"Submission UI dataset/domain/horizon/model combinations checked: {checked}")

    def test_evaluation_survives_rerun_for_multiple_downloads(self):
        app=AppTest.from_file(str(ROOT/"app"/"app.py"),default_timeout=90).run()
        next(b for b in app.button if b.label=="Run historical evaluation").click().run()
        self.assertTrue(any("Adaptation changes" in m.value for m in app.markdown))
        app.run()
        self.assertEqual(list(app.exception),[])
        self.assertTrue(any("Adaptation changes" in m.value for m in app.markdown))

    def test_prediction_type_switch_changes_forecast_shape(self):
        app=AppTest.from_file(str(ROOT/"app"/"app.py"),default_timeout=90).run()
        for ds in ("climate","energy"):
            app.selectbox(key="dataset").set_value(ds).run()
            for h in (1,24,1):
                app.radio(key=f"horizon_{ds}").set_value(h).run()
                app.checkbox[0].set_value(True).run()
                self.assertEqual(list(app.exception),[])
                # The visible forecast table must reflect the selected checkpoint horizon.
                forecast=app.dataframe[0].value
                self.assertEqual(forecast.shape,(h,7))
                self.assertTrue(np.isfinite(forecast.to_numpy()).all())
                self.assertEqual(len(app.metric),0)  # Beyond-file values have no actuals.
                self.assertTrue(any(("Next-hour forecast" if h==1 else "Next 24-hour forecast")==s.value for s in app.subheader))


if __name__=='__main__':unittest.main()
