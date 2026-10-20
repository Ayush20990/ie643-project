"""Verify decoder composition against its exact retained-adapted backbone."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import MODEL_DIR, PROC_DIR, RESULTS_DIR
from .coupled_transfer import FAMILY
from .improve import write_json
from .inference import load_coupled_bundle, forecast_batches
from .peak_data import prepare_peak
from .progressive_train import sha


def main():
    torch.set_num_threads(2)
    audits={}
    for ds in ("climate","energy"):
        out,dest=RESULTS_DIR/FAMILY/ds,MODEL_DIR/FAMILY/ds
        manifest=json.loads((out/"manifest.json").read_text())
        for name,value in manifest["code_sha256"].items():assert sha(Path(__file__).with_name(name))==value,name
        for name,value in manifest["data_sha256"].items():assert sha(PROC_DIR/ds/name)==value
        for name,value in manifest["initial_checkpoint_sha256"].items():assert sha(MODEL_DIR/name)==value
        freeze=json.loads((out/"selection_frozen.json").read_text())
        assert freeze["test_evaluated_at_freeze"] is False
        for name,value in freeze["checkpoint_sha256"].items():assert sha(dest/name)==value
        frame=pd.read_csv(out/"metrics_all.csv")
        assert len(frame)==24 and not frame.duplicated(["seed","domain","method"]).any()
        assert np.isfinite(frame.select_dtypes("number")).all().all()
        assert (out/"metrics_all.csv").stat().st_mtime>=pd.Timestamp(freeze["timestamp_utc"]).timestamp()
        choices=[]
        folder="peak_litetcn" if ds=="climate" else "residual_litetcn"
        for seed in (0,1,2):
            choice=json.loads((out/f"seed_{seed}_selection.json").read_text())
            assert choice["target_future_labels_used_for_training_or_selection"] is False
            history=choice["history"]
            selected=min((r for r in history if r["eligible"]),key=lambda r:r["proxy"])
            assert choice["chosen"]==selected["candidate"]
            assert np.isclose(choice["proxy"],selected["proxy"])
            for r in history:
                assert r["eligible"]==(r["source"]["MSE"]<=1.1*history[0]["source"]["MSE"])
                proxy=.5*r["source"]["score"]/history[0]["source"]["score"]+.5*r["target_input"]["score"]/history[0]["target_input"]["score"]
                assert np.isclose(proxy,r["proxy"])
            seed_dir=dest/f"seed_{seed}"
            base=torch.load(MODEL_DIR/folder/ds/f"seed_{seed}"/"adapted_model.pt",weights_only=True)
            source=torch.load(MODEL_DIR/"coupled_litetcn"/ds/f"seed_{seed}"/"source_model.pt",weights_only=True)
            current_source=torch.load(seed_dir/"source_model.pt",weights_only=True)
            assert all(torch.equal(v,source[n]) for n,v in current_source.items())
            for name in ("adapted_backbone_only","source_decoder_on_adapted_backbone","adapted_model"):
                state=torch.load(seed_dir/f"{name}.pt",weights_only=True)
                assert all(torch.equal(v,base[n.removeprefix('base.')]) for n,v in state.items() if n.startswith("base."))
            zero=torch.load(seed_dir/"adapted_backbone_only.pt",weights_only=True)
            assert zero["decoder.4.weight"].count_nonzero()==0 and zero["decoder.4.bias"].count_nonzero()==0
            if choice["chosen"] in ("zero_correction","source_decoder"):
                reference="adapted_backbone_only" if choice["chosen"]=="zero_correction" else "source_decoder_on_adapted_backbone"
                state=torch.load(seed_dir/"adapted_model.pt",weights_only=True)
                ref=torch.load(seed_dir/f"{reference}.pt",weights_only=True)
                assert all(torch.equal(v,ref[n]) for n,v in state.items())
            choices.append({"seed":seed,"chosen":choice["chosen"]})
        meta,models=load_coupled_bundle(ds,True)
        data=prepare_peak(ds);cfg=data["cfg"];key=cfg["columns"].index(cfg["key"])
        for role,name in (("source","source_model"),("target","adapted_model")):
            model=models[name];x,y=data[role]["test_packed"]
            p=forecast_batches(model,x)
            expected=frame.query("domain==@role and seed==0 and method==@name").MSE.item()
            assert np.isclose(np.mean((p-y.numpy())**2),expected,rtol=1e-5)
            with torch.no_grad():base=model.base(x[:64] if model.base_kind=="peak" else x[:64,...,:model.C]).numpy()
            keep=[c for c in range(model.C) if c!=key]
            np.testing.assert_array_equal(p[:64,...,keep],base[...,keep])
            assert sha(dest/f"{name}.pt")==sha(dest/"seed_0"/f"{name}.pt")
        audits[ds]={"code_data_initial_and_frozen_hashes":True,"selection_recomputed":True,
            "retained_adapted_backbone_unchanged":True,"non_temperature_forecasts_unchanged":True,
            "deployment_full_test_MSE_recomputed":True,"seeds":choices}
    write_json(RESULTS_DIR/FAMILY/"artifact_audit.json",audits)
    print(json.dumps(audits,indent=2))


if __name__=="__main__":main()
