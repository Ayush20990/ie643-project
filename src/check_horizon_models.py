"""Audit next-hour and peak-aware 24-hour models without changing selection."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import MODEL_DIR,PROC_DIR,RESULTS_DIR
from .improve import write_json
from .inference import load_event_boost_bundle,load_next_hour_bundle,load_peak_next_hour_bundle,forecast_batches
from .peak_data import prepare_peak
from .progressive_train import sha


def audit_hashes(family,ds):
    out,dest=RESULTS_DIR/family/ds,MODEL_DIR/family/ds
    manifest=json.loads((out/"manifest.json").read_text())
    for name,v in manifest["code_sha256"].items():assert sha(Path(__file__).with_name(name))==v,(family,name)
    for name,v in manifest["data_sha256"].items():assert sha(PROC_DIR/ds/name)==v
    for collection in ("base_sha256","initial_checkpoint_sha256","expert_sha256"):
        for name,v in manifest.get(collection,{}).items():assert sha(MODEL_DIR/name)==v
    freeze=json.loads((out/"selection_frozen.json").read_text())
    assert freeze["test_evaluated_at_freeze"] is False
    for name,v in freeze.get("checkpoint_sha256",freeze.get("model_sha256",{})).items():assert sha(dest/name)==v
    assert (out/"metrics_all.csv").stat().st_mtime>=pd.Timestamp(freeze["timestamp_utc"]).timestamp()
    return manifest


def main():
    torch.set_num_threads(2)
    audits,timings={},[]
    for ds in ("climate","energy"):
        data=prepare_peak(ds);cfg=data["cfg"]
        for family in ("next_hour","next_hour_hybrid","event_boost","peak_next_hour"):
            manifest=audit_hashes(family,ds)
            out=RESULTS_DIR/family/ds;dest=MODEL_DIR/family/ds
            f=pd.read_csv(out/"metrics_all.csv")
            assert not f.duplicated(["seed","domain","method"]).any()
            assert np.isfinite(f.select_dtypes("number")).all().all()
            for seed in (0,1,2):
                record=json.loads((out/f"seed_{seed}_selection.json").read_text())
                assert record["target_future_labels_used_for_training_or_selection"] is False
                if family=="next_hour":
                    scores={}
                    for mode in ("uniform","turning"):
                        r=json.loads((out/f"seed_{seed}_{mode}.json").read_text())
                        best=min((v for v in r["history"] if v["eligible"]),key=lambda v:v["score"])
                        assert r["selected_round"]==best["round"]
                        scores[mode]=r["score"]
                    assert record["source_control"]==min(scores,key=scores.get)
                    h=json.loads((out/f"seed_{seed}_adaptation.json").read_text())["history"]
                    best=min((v for v in h if v["eligible"]),key=lambda v:v["proxy"])
                    assert record["adaptation_round"]==best["round"]
                    for v in h:
                        expected=.5*v["source"]["score"]/h[0]["source"]["score"]+.5*v["target_input"]["score"]/h[0]["target_input"]["score"]
                        assert np.isclose(v["proxy"],expected)
                        assert v["eligible"]==(v["source"]["MSE"]<=1.1*h[0]["source"]["MSE"] and v["source"]["key_MSE"]<=1.1*h[0]["source"]["key_MSE"])
                elif family=="next_hour_hybrid":
                    for v in record["calibration"]:
                        assert np.isclose(sum(v["source_weights"]),1.) and np.isclose(sum(v["adapted_weights"]),1.)
                        assert min(v["source_weights"]+v["adapted_weights"])>=0.
                        assert v["source_weights"][1]==0. and v["source_weights"][3]==0.
                        assert v["adapted_source_MSE"]<=1.1*v["source_reference_MSE"]+1e-8
                elif family=="peak_next_hour":
                    for v in record["calibration"]:
                        sw,aw=np.asarray(v["source_weights"]),np.asarray(v["adapted_weights"])
                        np.testing.assert_allclose(sw.sum(1),1.);np.testing.assert_allclose(aw.sum(1),1.)
                        assert min(sw.min(),aw.min())>=0. and not sw[:,[1,3]].any()
                        assert v["adapted_source_point"]<=1.1*v["source_point_reference"]+1e-8
                        assert v["adapted_target_input_point"]<=1.05*v["target_input_point_reference"]+1e-8
                        expected=(.5*v["adapted_source_event"]/v["source_event_reference"]+
                                  .5*v["adapted_target_input_event"]/v["target_input_event_reference"]+
                                  .001*np.sum((aw-sw)**2))
                        assert np.isclose(v["proxy"],expected) and v["proxy"]<=1.+1e-8
                        if not v["accepted"]:np.testing.assert_array_equal(sw,aw)
                        sparse=np.minimum(v["source_regime_counts"],v["target_input_regime_counts"])<64
                        np.testing.assert_array_equal(sw[sparse],aw[sparse])
                else:
                    source=json.loads((out/f"seed_{seed}_source_search.json").read_text())
                    adapted=json.loads((out/f"seed_{seed}_adaptation_search.json").read_text())
                    best_source=min((v for v in source if v["eligible"]),key=lambda v:v["quality"]["score"])
                    best_adapted=min((v for v in adapted if v["eligible"]),key=lambda v:v["proxy"])
                    assert record["source"]==best_source
                    assert record["adapted"]==best_adapted
                    for v in source:assert v["eligible"]==(v["quality"]["MSE"]<=1.05*source[0]["quality"]["MSE"])
                    for v in adapted:
                        assert v["eligible"]==(v["source"]["MSE"]<=1.05*adapted[0]["source"]["MSE"])
                        expected=.5*v["source"]["score"]/adapted[0]["source"]["score"]+.5*v["target_input"]["score"]/adapted[0]["target_input"]["score"]
                        assert np.isclose(v["proxy"],expected)
            if family=="event_boost":meta,models=load_event_boost_bundle(ds)
            elif family=="peak_next_hour":meta,models=load_peak_next_hour_bundle(ds)
            else:meta,models=load_next_hour_bundle(ds,family=="next_hour_hybrid")
            for role,name in (("source","source_model"),("target","adapted_model")):
                model=models[name];x,y=data[role]["test_packed"]
                p=forecast_batches(model,x)
                truth=y.numpy()[:,:meta["H"]]
                assert p.shape==truth.shape
                actual=np.mean((p-truth)**2)
                expected=f.query("seed==0 and domain==@role and method==@name").MSE.item()
                assert np.isclose(actual,expected,rtol=1e-5),(family,ds,actual,expected)
                if family=="event_boost":
                    with torch.no_grad():base=model.base_prediction(x[:64]).numpy()
                    key=cfg["columns"].index(cfg["key"]);keep=[j for j in range(model.C) if j!=key]
                    np.testing.assert_array_equal(base[...,keep],p[:64,...,keep])
                    if model.selection["weight"]:assert model.tree.tree_count_==model.selection["trees"]
            model=models["adapted_model"];sample=data["target"]["test_packed"][0][:1]
            torch.set_num_threads(1)
            with torch.no_grad():
                for _ in range(5):model(sample)
                started=time.perf_counter()
                for _ in range(30):model(sample)
            elapsed_ms=(time.perf_counter()-started)/30*1000
            deployed_KB=((dest/"seed_0"/"adapted_backbone.pt").stat().st_size+
                         ((dest/"seed_0"/"adapted_tree.cbm").stat().st_size if model.selection["weight"] else 0))/1024 if family=="event_boost" else (dest/"seed_0"/"adapted_model.pt").stat().st_size/1024
            timings.append({"dataset":ds,"family":family,"horizon":meta["H"],
                "neural_parameters":sum(p.numel() for p in model.parameters()),
                "stored_tensor_values":sum(v.numel() for v in model.state_dict().values()),
                "boosted_leaf_values":int(model.tree.get_leaf_values().size) if family=="event_boost" and model.selection["weight"] else 0,
                "adapted_forecaster_KB":deployed_KB,
                "seed0_artifact_KB_including_controls":sum(p.stat().st_size for p in (dest/"seed_0").glob("*"))/1024,
                "batch_1_CPU_ms":elapsed_ms,"torch_threads":1,
                "native_tree_threads":2 if family=="event_boost" and model.selection["weight"] else 0})
            torch.set_num_threads(2)
            audits[f"{family}/{ds}"]={"code_data_initial_and_frozen_hashes":True,"selection_recomputed":True,
                "source_and_target_full_seed0_MSE_recomputed":True,"forecast_horizon":meta["H"]}
    dest=RESULTS_DIR/"forecast_horizons";dest.mkdir(exist_ok=True)
    write_json(dest/"artifact_audit.json",audits)
    pd.DataFrame(timings).to_csv(dest/"efficiency.csv",index=False)
    print(json.dumps(audits,indent=2))


if __name__=="__main__":main()
