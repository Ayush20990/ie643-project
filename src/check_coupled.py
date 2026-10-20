"""Audit predicted-driver control selection and frozen LiteTCN preservation."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import MODEL_DIR, PROC_DIR, RESULTS_DIR
from .coupled_train import FAMILY, CONTROLS
from .improve import write_json
from .inference import load_coupled_bundle, forecast_batches
from .peak_data import prepare_peak
from .progressive_train import sha


def main():
    torch.set_num_threads(2)
    audits,efficiency={},[]
    for ds in ("climate","energy"):
        out,dest=RESULTS_DIR/FAMILY/ds,MODEL_DIR/FAMILY/ds
        manifest=json.loads((out/"manifest.json").read_text())
        for name,value in manifest["code_sha256"].items():assert sha(Path(__file__).with_name(name))==value,name
        for name,value in manifest["data_sha256"].items():assert sha(PROC_DIR/ds/name)==value
        for name,value in manifest["base_sha256"].items():assert sha(MODEL_DIR/name)==value
        freeze=json.loads((out/"selection_frozen.json").read_text())
        assert freeze["test_evaluated_at_freeze"] is False
        for name,value in freeze["checkpoint_sha256"].items():assert sha(dest/name)==value
        frame=pd.read_csv(out/"metrics_all.csv")
        assert len(frame)==24 and not frame.duplicated(["seed","domain","method"]).any()
        assert np.isfinite(frame.select_dtypes("number")).all().all()
        assert (out/"metrics_all.csv").stat().st_mtime>=pd.Timestamp(freeze["timestamp_utc"]).timestamp()
        choices=[]
        for seed in (0,1,2):
            selection=json.loads((out/f"seed_{seed}_selection.json").read_text())
            assert selection["target_future_labels_used_for_training_or_selection"] is False
            scores={}
            for control in CONTROLS:
                record=json.loads((out/f"seed_{seed}_{control}.json").read_text())
                best=min((r for r in record["history"] if r["eligible"]),key=lambda r:r["proxy"])
                assert best["round"]==record["selected_round"]
                assert np.isclose(best["proxy"],record["score"])
                scores[control]=record["score"]
                for r in record["history"]:
                    assert r["eligible"]==(r["source"]["MSE"]<=1.03*record["history"][0]["source"]["MSE"])
            chosen=min(scores,key=scores.get)
            assert selection["source_control"]==chosen
            seed_dir=dest/f"seed_{seed}"
            source=torch.load(seed_dir/"source_model.pt",weights_only=True)
            picked=torch.load(seed_dir/f"{chosen}.pt",weights_only=True)
            assert all(torch.equal(v,picked[k]) for k,v in source.items())
            record=json.loads((out/f"seed_{seed}_adaptation.json").read_text())
            history=record["history"]
            best=min((r for r in history if r["eligible"]),key=lambda r:r["proxy"])
            assert selection["adaptation_round"]==best["round"]
            for r in history:
                assert r["eligible"]==(r["source"]["MSE"]<=1.1*history[0]["source"]["MSE"])
                proxy=.5*r["source"]["score"]/history[0]["source"]["score"]+.5*r["target_input"]["score"]/history[0]["target_input"]["score"]
                assert np.isclose(proxy,r["proxy"])
            folder="peak_litetcn" if ds=="climate" else "residual_litetcn"
            base=torch.load(MODEL_DIR/folder/ds/f"seed_{seed}"/"source_model.pt",weights_only=True)
            for name in (*CONTROLS,"source_model","adapted_model"):
                weights=torch.load(seed_dir/f"{name}.pt",weights_only=True)
                assert all(torch.equal(v,base[k.removeprefix('base.')]) for k,v in weights.items() if k.startswith("base."))
            choices.append({"seed":seed,"source_control":chosen,"adaptation_round":selection["adaptation_round"]})
        meta,models=load_coupled_bundle(ds)
        data=prepare_peak(ds);cfg=data["cfg"];key=cfg["columns"].index(cfg["key"])
        for role,name in (("source","source_model"),("target","adapted_model")):
            x,y=data[role]["test_packed"]
            p=forecast_batches(models[name],x)
            expected=frame.query("domain==@role and seed==0 and method==@name").MSE.item()
            assert np.isclose(float(np.mean((p-y.numpy())**2)),expected,rtol=1e-5)
            model=models[name]
            with torch.no_grad():
                xb=x[:64]
                base=model.base(xb if model.base_kind=="peak" else xb[...,:model.C]).numpy()
            keep=[c for c in range(model.C) if c!=key]
            np.testing.assert_array_equal(p[:64,...,keep],base[...,keep])
            assert sha(dest/f"{name}.pt")==sha(dest/"seed_0"/f"{name}.pt")
        model=models["adapted_model"];sample=data["target"]["test_packed"][0][:1]
        torch.set_num_threads(1)
        with torch.no_grad():
            for _ in range(5):model(sample)
            started=time.perf_counter()
            for _ in range(30):model(sample)
        efficiency.append({"dataset":ds,"parameters":sum(p.numel() for p in model.parameters()),
            "updated_parameters":sum(p.numel() for p in model.decoder.parameters()),"batch_1_CPU_ms":(time.perf_counter()-started)/30*1000})
        torch.set_num_threads(2)
        audits[ds]={"code_data_base_checkpoint_hashes":True,"selection_recomputed":True,"base_weights_unchanged":True,
            "other_channel_forecasts_unchanged":True,"full_deployment_test_MSE_recomputed":True,"seeds":choices}
    write_json(RESULTS_DIR/FAMILY/"artifact_audit.json",audits)
    pd.DataFrame(efficiency).to_csv(RESULTS_DIR/FAMILY/"efficiency.csv",index=False)
    print(json.dumps(audits,indent=2))


if __name__=="__main__":main()
