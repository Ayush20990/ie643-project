"""App smoke checks using the saved checkpoints and bundled samples."""
import unittest
from unittest.mock import patch

import torch
from streamlit.testing.v1 import AppTest

from src.config import DATASETS, ROOT
from src.inference import read_forecast_csv


class AppSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_both_domains_and_all_datasets(self):
        app = AppTest.from_file(str(ROOT / "app" / "research.py"), default_timeout=60).run()
        checked_combinations = 0
        for ds in DATASETS:
            app.selectbox[0].set_value(ds).run()
            for family in list(app.selectbox[1].options):
                app.selectbox[1].set_value(family).run()
                for domain in ("Domain 1 (source)", "Domain 2 (target)"):
                    with self.subTest(dataset=ds, experiment=family, domain=domain):
                        app.radio[0].set_value(domain).run()
                        self.assertEqual(list(app.exception), [])
                        self.assertEqual(len(app.metric), 4)
                        checked_combinations += 1
        print(f"UI combinations checked: {checked_combinations}")

    def test_exact_lookback_rows_forecast_without_future_actuals(self):
        def short_file(file, ds, cols):
            return read_forecast_csv(file, ds, cols).iloc[:DATASETS[ds]["L"]]
        with patch("src.inference.read_forecast_csv", side_effect=short_file):
            app = AppTest.from_file(str(ROOT / "app" / "research.py"), default_timeout=60).run()
            self.assertEqual(list(app.exception), [])
            self.assertEqual(len(app.metric), 0)
            self.assertTrue(any("no actual values" in info.value for info in app.info))

    def test_next_hour_comparisons_do_not_broadcast_24h_predictions(self):
        from src.metrics import all_metrics
        if not (ROOT / "models" / "next_hour_hybrid" / "climate" / "meta.json").exists():
            self.skipTest("Next-hour checkpoints not trained yet")
        seen = []
        def checked(x, y, p, *args, **kwargs):
            self.assertEqual(y.shape, p.shape, "Baseline comparisons must score the same forecast horizon")
            seen.append(y.shape[1])
            return all_metrics(x, y, p, *args, **kwargs)
        with patch("src.metrics.all_metrics", side_effect=checked):
            app = AppTest.from_file(str(ROOT / "app" / "research.py"), default_timeout=60).run()
            app.selectbox[0].set_value("climate").run()
            app.selectbox[1].set_value("Next-hour LiteTCN + local dynamics").run()
            self.assertEqual(list(app.exception), [])
            self.assertTrue(any("history is updated hourly" in c.value for c in app.caption))
        self.assertIn(1, seen)


if __name__ == "__main__":
    unittest.main()
