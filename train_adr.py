"""SLAI v2.3 DelegationRanker CLI. Offline, provenance-gated model training.

Examples:
 python -m train_adr data-audit
 python -m train_adr prepare-data
 python -m train_adr train --epochs 40
 python -m train_adr evaluate
 python -m train_adr smoke   # ONLY synthetic internal test data
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import time
import uuid

from src.agents.collaborative.delegation_ranker import (SCHEMA, INPUT_DIM, DelegationEventStore,
                                                       DelegationRanker, _atomic_json)

DEFAULT_DATA = Path("data/training/collaborative/delegation_ranker")
DEFAULT_RUN = DEFAULT_DATA / "run_v1"
DEFAULT_MODEL = Path("src/agents/collaborative/models/delegation_ranker_v1")


def atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        temp.write_bytes(content)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def safe_torch_save(value, path: Path) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        torch.save(value, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(data_dir: Path) -> dict:
    """Extract ONLY observed success/failure from durable router evidence."""
    store = DelegationEventStore(data_dir / "raw" / "events.sqlite3")
    rows, counts = store.samples()
    destination = data_dir / "validated" / "attempts.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temp.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
        os.replace(temp, destination)
    finally:
        temp.unlink(missing_ok=True)
    manifest = {"schema": SCHEMA, "source": "slai_task_router_attempt", "origin": str(store.path),
                "samples": len(rows), "positive": sum(x["success"] for x in rows),
                "negative": sum(not x["success"] for x in rows),
                "other_outcomes": counts, "sha256": digest_file(destination),
                "synthetic": False, "generated_at": time.time()}
    _atomic_json(data_dir / "validated" / "dataset_manifest.json", manifest)
    return manifest


def read_validated(data_dir: Path, *, allow_synthetic: bool = False):
    file = data_dir / "validated" / "attempts.jsonl"
    manifest_file = data_dir / "validated" / "dataset_manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or digest_file(file) != manifest.get("sha256"):
        raise ValueError("Dataset hash/schema mismatch. Re-run prepare-data.")
    if manifest.get("synthetic") and not allow_synthetic:
        raise ValueError("Synthetic fixtures cannot be used to promote a production model.")
    rows = []
    with file.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            row = json.loads(line)
            if not isinstance(row.get("x"), list) or len(row["x"]) != INPUT_DIM:
                raise ValueError(f"Bad feature width at dataset line {i}")
            if row.get("success") not in (0, 1) or row.get("provenance") != "slai_task_router_attempt" and not manifest.get("synthetic"):
                raise ValueError(f"Invalid observed outcome/provenance at line {i}")
            if not all(isinstance(z, (int, float)) and math.isfinite(float(z)) for z in row["x"]):
                raise ValueError(f"Nonfinite features at line {i}")
            if row.get("latency_ms") is not None and (not isinstance(row["latency_ms"], (float, int)) or row["latency_ms"] < 0):
                raise ValueError(f"Invalid latency at line {i}")
            rows.append(row)
    if len(rows) != manifest.get("samples"):
        raise ValueError("Dataset manifest count mismatch")
    return rows, manifest


def split_by_task_time(rows):
    """Time ordered task groups, no task ID overlaps across partitions."""
    groups = defaultdict(list)
    for r in rows:
        groups[str(r["task_id"])].append(r)
    ordered = sorted(groups.items(), key=lambda item: (min(x["timestamp"] for x in item[1]), item[0]))
    if len(ordered) < 9:
        raise ValueError("Need >=9 distinct task groups for time-grouped train/validation/test evaluation")
    n = len(ordered)
    a = max(1, int(n * 0.70))
    b = max(a + 1, int(n * 0.85))
    if b >= n:
        b = n - 1
    partitions = [ordered[:a], ordered[a:b], ordered[b:]]
    return [[r for _, items in p for r in items] for p in partitions]


def stats(rows):
    import numpy as np
    m = np.asarray([r["x"] for r in rows], dtype="float32")
    mean = m.mean(axis=0)
    scale = m.std(axis=0)
    scale = np.maximum(scale, 0.05)
    return mean.tolist(), scale.tolist()


def build_net(width: int):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(INPUT_DIM, width), nn.ReLU(), nn.Linear(width, 2))


def metrics(probs: list[float], labels: list[int]) -> dict:
    if not labels:
        return {"n": 0}
    brier = sum((p-y)**2 for p,y in zip(probs,labels)) / len(labels)
    positives = sum(labels)
    bins = [[] for _ in range(10)]
    for p, y in zip(probs, labels):
        bins[min(9, int(max(0, min(1, p))*10))].append((p,y))
    ece = sum(len(b)/len(labels) * abs(sum(p for p,_ in b)/len(b) - sum(y for _,y in b)/len(b)) for b in bins if b)
    # AUC computed by pairwise concordance, memory bounded; O(N log N) via sorted groups.
    sorted_rows = sorted(zip(probs,labels))
    neg_before = 0
    concordant = 0.0
    i=0
    while i < len(sorted_rows):
        j=i
        while j<len(sorted_rows) and sorted_rows[j][0]==sorted_rows[i][0]:
            j+=1
        ys = [y for _,y in sorted_rows[i:j]]
        ps=sum(ys); ns=len(ys)-ps
        concordant += ps*neg_before + 0.5*ps*ns
        neg_before += ns
        i=j
    auc = concordant/(positives*(len(labels)-positives)) if 0<positives<len(labels) else None
    return {"n": len(labels), "positive": positives, "brier": brier, "ece_10": ece,
            "roc_auc": auc, "accuracy_threshold_0_5": sum((p>=0.5)==bool(y) for p,y in zip(probs,labels))/len(labels)}


def infer(net, rows, mean, scale, device):
    import torch
    net.eval()
    probs=[]
    with torch.inference_mode():
        for at in range(0,len(rows),1024):
            chunk=rows[at:at+1024]
            x=torch.tensor([[(v-m)/s for v,m,s in zip(r["x"], mean,scale)] for r in chunk],dtype=torch.float32,device=device)
            probs.extend(torch.sigmoid(net(x)[:,0]).cpu().tolist())
    return probs


def run_train(args, *, allow_synthetic=False):
    import torch
    import torch.nn.functional as F
    data_dir=Path(args.data_dir); run_dir=Path(args.run_dir); model_dir=Path(args.model_dir)
    rows, source = read_validated(data_dir, allow_synthetic=allow_synthetic)
    train,valid,test=split_by_task_time(rows)
    if len(train)<20 or len(valid)<3 or len(test)<3 or min(sum(r["success"] for r in train),sum(not r["success"] for r in train))<4:
        raise ValueError("Insufficient verified and diverse observations: >=20 train, >=3 validation/test, >=4 samples of both labels in train required")
    if args.resume and not (run_dir/"latest.state.pt").exists():
        raise FileNotFoundError("No last optimizer state for resume")
    if not args.resume and any((run_dir/name).exists() for name in ("latest.state.pt","best.pt")):
        raise FileExistsError("Training run exists; use --resume or a new --run-dir")
    if not args.resume and model_dir.exists() and any(model_dir.iterdir()):
        raise FileExistsError("Export directory already populated; choose new --model-dir")
    torch.manual_seed(args.seed); random.seed(args.seed)
    if torch.cuda.is_available() and args.device=="cuda":
        device="cuda"
    elif args.device=="cuda":
        raise ValueError("CUDA requested but unavailable")
    else:
        device="cpu"
    mean,scale=stats(train)
    net=build_net(args.width).to(device)
    opt=torch.optim.AdamW(net.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="min",patience=3,factor=0.5)
    start=0; best=float("inf"); stale=0
    dataset_hash=source["sha256"]
    if args.resume:
        state=torch.load(run_dir/"latest.state.pt",map_location="cpu",weights_only=True)
        if state["dataset_hash"]!=dataset_hash or state["width"]!=args.width or state["seed"]!=args.seed:
            raise ValueError("Incompatible resume dataset/architecture/seed")
        net.load_state_dict(state["model"]); opt.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["torch_rng"]); random.setstate(json_to_tuple(state["python_rng"]))
        mean,scale=state["mean"],state["scale"]
        start=state["epoch"]+1; best=state["best"]; stale=state["stale"]
    run_dir.mkdir(parents=True,exist_ok=True)
    _atomic_json(run_dir/"training_config.json",vars(args) | {"device_resolved":device,"dataset_hash":dataset_hash})
    train_x=torch.tensor([[(v-m)/s for v,m,s in zip(r["x"],mean,scale)] for r in train],dtype=torch.float32)
    train_y=torch.tensor([r["success"] for r in train],dtype=torch.float32)
    pos=train_y.sum().item(); neg=len(train)-pos
    positive_weight=torch.tensor(min(10.0,neg/max(1.0,pos)),dtype=torch.float32,device=device)
    latency_target=torch.tensor([math.log1p(max(0,r.get("latency_ms") or 0)) for r in train],dtype=torch.float32)
    latency_mask=torch.tensor([r.get("latency_ms") is not None and r["success"]==1 for r in train],dtype=torch.bool)
    has_latency=bool(latency_mask.sum().item()>=15)
    started=time.monotonic()
    for epoch in range(start,args.epochs):
        net.train()
        idx=torch.randperm(len(train))
        cumulative=0.0
        for chunk in idx.split(args.batch_size):
            x=train_x[chunk].to(device); y=train_y[chunk].to(device)
            opt.zero_grad(set_to_none=True)
            out=net(x)
            loss=F.binary_cross_entropy_with_logits(out[:,0],y,pos_weight=positive_weight)
            mask=latency_mask[chunk].to(device)
            if has_latency and mask.any():
                loss=loss+0.05*F.smooth_l1_loss(out[mask,1],latency_target[chunk].to(device)[mask])
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.0)
            opt.step()
            cumulative += float(loss.item())*len(chunk)
        probs=infer(net,valid,mean,scale,device)
        # Validation Brier for calibrated success predictions; separate from weighted BCE.
        vm=metrics(probs,[r["success"] for r in valid]); val=vm["brier"]
        scheduler.step(val)
        improved=val<best-1e-7
        if improved:
            best=val;stale=0
            safe_torch_save({k:v.detach().cpu() for k,v in net.state_dict().items()},run_dir/"best.pt")
        else: stale+=1
        state={"model":{k:v.detach().cpu() for k,v in net.state_dict().items()},
               "optimizer":opt.state_dict(),"scheduler":scheduler.state_dict(),
               "torch_rng":torch.get_rng_state(),"python_rng":listify(random.getstate()),
               "epoch":epoch,"width":args.width,"seed":args.seed,"mean":mean,"scale":scale,
               "best":best,"stale":stale,"dataset_hash":dataset_hash}
        safe_torch_save(state,run_dir/"latest.state.pt")
        with (run_dir/"training_history.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps({"epoch":epoch,"training_loss":cumulative/len(train),
                                "validation":vm,"lr":opt.param_groups[0]["lr"],"elapsed_s":time.monotonic()-started})+"\n")
        if stale>=args.patience:
            break
    if not (run_dir/"best.pt").exists():
        raise RuntimeError("No validated best checkpoint")
    net.load_state_dict(torch.load(run_dir/"best.pt",map_location="cpu",weights_only=True))
    test_metrics=metrics(infer(net,test,mean,scale,device),[r["success"] for r in test])
    train_rate=sum(r["success"] for r in train)/len(train)
    baseline_brier=sum((train_rate-r["success"])**2 for r in test)/len(test)
    report={"model_selected_by":"validation_brier","test":test_metrics,
            "constant_train_rate_baseline_brier":baseline_brier,
            "train_n":len(train),"validation_n":len(valid),"test_n":len(test),
            "distinct_task_groups":[len(set(x["task_id"] for x in part)) for part in (train,valid,test)],
            "dataset_sha256":dataset_hash,"source":source["source"],"synthetic":source["synthetic"],
            "off_policy_caveat":"Outcomes exist only for attempted agents. These metrics cannot establish routing policy improvement.",
            "deployment_approved":False,"wall_time_seconds":time.monotonic()-started}
    _atomic_json(run_dir/"evaluation.json",report)
    if not source["synthetic"]:
        model_dir.mkdir(parents=True,exist_ok=True)
        atomic_bytes(model_dir/"model.pt",(run_dir/"best.pt").read_bytes())
        _atomic_json(model_dir/"features.json",{"schema":SCHEMA,"input_dim":INPUT_DIM,"mean":mean,"scale":scale,
                                                   "hash_buckets":16,"feature_contract":"pre-decision-only"})
        _atomic_json(model_dir/"manifest.json",{"schema":SCHEMA,"input_dim":INPUT_DIM,"width":args.width,
                       "version":"delegation_ranker_v1","has_latency":has_latency,
                       "dataset_sha256":dataset_hash,"sha256":digest_file(model_dir/"model.pt"),
                       "deployment_approved":False})
        _atomic_json(model_dir/"evaluation.json",report)
        _atomic_json(model_dir/"training_config.json",vars(args))
        DelegationRanker(model_dir).predict([rows[0]["x"]])
    return report


def listify(v):
    return [listify(x) for x in v] if isinstance(v,tuple) else v


def json_to_tuple(v):
    return tuple(json_to_tuple(x) for x in v) if isinstance(v,list) else v


def evaluate(args):
    # Recomputes held-out evaluation from the persisted model; refuses dataset changes.
    d=Path(args.model_dir)
    model=DelegationRanker(d)
    manifest=json.loads((d/"manifest.json").read_text(encoding="utf-8"))
    rows,source=read_validated(Path(args.data_dir))
    if source["sha256"]!=manifest["dataset_sha256"]:
        raise ValueError("Saved model and evaluation dataset hashes do not match")
    train,valid,test=split_by_task_time(rows)
    start=time.perf_counter()
    probs=[float(x["success_probability"]) for x in model.predict([r["x"] for r in test])]
    runtime=(time.perf_counter()-start)*1000/max(1,len(test))
    observed=metrics(probs,[r["success"] for r in test])
    rate=sum(r["success"] for r in train)/len(train)
    return {"checkpoint_verified":True,"version":model.version,"held_out":observed,
            "predict_ms_per_pair":runtime,"baseline_constant_brier":
            sum((rate-r["success"])**2 for r in test)/len(test),
            "counterfactual_routing_improvement_established":False}


def smoke(args):
    # Explicitly synthetic data; never exported to a production checkpoint.
    with tempfile.TemporaryDirectory(prefix="adr-smoke-") as temp:
        d=Path(temp); target=d/"validated";target.mkdir()
        rng=random.Random(19)
        rows=[]
        for i in range(120):
            x=[0.0]*INPUT_DIM; x[0]=1.0; x[1]=float(i%2); x[3]=rng.random()
            rows.append({"decision_id":str(i),"task_id":str(i),"timestamp":i,
                         "agent_id":"smoke","x":x,"success":int(x[1]>0),
                         "latency_ms":float(10+i%3),"provenance":"synthetic_smoke_fixture"})
        file=target/"attempts.jsonl"
        file.write_text("".join(json.dumps(r)+"\n" for r in rows),encoding="utf-8")
        _atomic_json(target/"dataset_manifest.json",{"schema":SCHEMA,"samples":len(rows),
                     "sha256":digest_file(file),"synthetic":True,"source":"synthetic_smoke_fixture"})
        args.data_dir=str(d);args.run_dir=str(d/"run");args.model_dir=str(d/"model")
        args.epochs=max(args.epochs,10)
        import torch
        torch.manual_seed(args.seed)
        before={k:v.detach().clone() for k,v in build_net(args.width).state_dict().items()}
        result=run_train(args,allow_synthetic=True)
        learned=torch.load(Path(args.run_dir)/"best.pt",map_location="cpu",weights_only=True)
        delta=sum(float((learned[k]-before[k]).abs().sum()) for k in before)
        if delta<=0 or not math.isfinite(result["test"]["brier"]):
            raise AssertionError("Smoke model failed to optimize parameters")
        # Load an actual trained checkpoint in a fresh inference object.
        training_state=torch.load(Path(args.run_dir)/"latest.state.pt",map_location="cpu",weights_only=True)
        m=Path(args.model_dir);m.mkdir(parents=True)
        atomic_bytes(m/"model.pt",(Path(args.run_dir)/"best.pt").read_bytes())
        _atomic_json(m/"features.json",{"mean":training_state["mean"],"scale":training_state["scale"]})
        _atomic_json(m/"manifest.json",{"schema":SCHEMA,"input_dim":INPUT_DIM,"width":args.width,
                    "version":"synthetic_smoke_only","has_latency":True,"sha256":digest_file(m/"model.pt"),
                    "deployment_approved":False,"synthetic":True})
        fresh=DelegationRanker(m)
        prediction=fresh.predict([rows[0]["x"]])[0]
        if not 0.0<=float(prediction["success_probability"])<=1.0:
            raise AssertionError("Reloaded inference violated probability contract")
        return {"smoke":"passed","learned_parameter_l1_change":delta,"reload_prediction":prediction,
                "train_samples":result["train_n"],
                "synthetic_data_only":True,"not_production_evidence":True}


def main(argv=None):
    p=argparse.ArgumentParser(description="SLAI Agent Delegation Ranker training and data audit")
    p.add_argument("command",choices=["status","data-audit","prepare-data","collect","train","resume","evaluate","export","smoke"])
    p.add_argument("--data-dir",default=str(DEFAULT_DATA))
    p.add_argument("--run-dir",default=str(DEFAULT_RUN))
    p.add_argument("--model-dir",default=str(DEFAULT_MODEL))
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--epochs",type=int,default=40)
    p.add_argument("--batch-size",type=int,default=64)
    p.add_argument("--width",type=int,default=32)
    p.add_argument("--lr",type=float,default=1e-3)
    p.add_argument("--weight-decay",type=float,default=1e-4)
    p.add_argument("--patience",type=int,default=8)
    p.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    p.add_argument("--resume",action="store_true")
    args=p.parse_args(argv)
    if args.epochs<=0 or args.batch_size<=0 or args.width<=0 or args.patience<=0 or args.lr<=0:
        p.error("All training sizes, patience and learning rate must be positive")
    try:
        if args.command in {"status","data-audit","collect"}:
            store=DelegationEventStore(Path(args.data_dir)/"raw"/"events.sqlite3")
            result={"data":store.audit(),"model_present":(Path(args.model_dir)/"manifest.json").exists()}
            if args.command=="collect":
                result["instruction"]="Enable delegation_ranker.mode=shadow in collaborative_config.yaml; actual routes produce SQLite observations"
        elif args.command=="prepare-data":result=prepare(Path(args.data_dir))
        elif args.command in {"train","resume"}:
            args.resume=args.resume or args.command=="resume";result=run_train(args)
        elif args.command=="smoke":result=smoke(args)
        elif args.command=="evaluate":result=evaluate(args)
        elif args.command=="export":
            raise ValueError("Export is performed by train; specify --model-dir before training a fresh version")
        print(json.dumps(result,indent=2))
        return 0
    except (ValueError,FileNotFoundError,FileExistsError,FloatingPointError,RuntimeError,ImportError) as exc:
        print("ADR ERROR:",exc,file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
