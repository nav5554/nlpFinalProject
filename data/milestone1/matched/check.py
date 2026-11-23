"""Recount held-out hits, and break them down by the reference token's surface text."""
from pathlib import Path
from collections import Counter
import json
import hashlib
import numpy as np
import torch
from transformers import AutoTokenizer

base = Path(__file__).resolve().parent
pred = json.loads((base / "predictions.json").read_text())
results = json.loads((base / "results.json").read_text())
sample = json.loads((base / "evaluation_sample.json").read_text())
tune = json.loads((base / "tuning_sample.json").read_text())
previous = json.loads((base.parent / "sentence/sample.json").read_text())
previous_windows = json.loads((base.parent / "run/windows.json").read_text())
assert not {r["article"] for r in pred} & {r["article"] for r in tune + previous}
starts = sorted(r["window_start"] for r in pred)
assert all(b-a >= 256 for a,b in zip(starts,starts[1:]))
assert all(abs(s - w*256) >= 256 for s in starts for w in previous_windows["validation_for_decoding"] + previous_windows["validation_for_training_and_AUROC"])
tok = AutoTokenizer.from_pretrained("gpt2",local_files_only=True)
group_names = ["contains_letter_or_digit","other"]
counts = Counter()
hits = {g:Counter() for g in group_names}
examples=[]
for row,src in zip(pred,sample,strict=True):
    assert all(row[key]==value for key,value in src.items())
    ref = row["reference_ids"]
    assert len(ref)==row["gap_length"]
    assert tok.decode(ref).strip()==row["sentence"].strip()
    groups = [group_names[0] if any(ch.isalnum() for ch in tok.decode([token])) else group_names[1] for token in ref]
    counts.update(groups)
    for method,ids in row["fill_ids"].items():
        assert len(ids)==len(ref) and 50257 not in ids
        assert row["correct"][method]==sum(a==b for a,b in zip(ids,ref,strict=True))
        for group,a,b in zip(groups,ids,ref,strict=True):
            hits[group][method] += int(a==b)
    # First reference word token in evaluation order recovered by one policy only.
    for i,(a,b,c) in enumerate(zip(ref,row["fill_ids"]["confidence"],row["fill_ids"]["mlp"],strict=True)):
        if (a==b)!=(a==c) and len(tok.decode([a]).strip())>=3 and groups[i]==group_names[0]:
            low,high=max(0,i-2),min(len(ref),i+3)
            examples.append({"row":row["source_row"],"offset":i,"token":tok.decode([a]),"reference":tok.decode(ref[low:high]),
                             "confidence":tok.decode(row["fill_ids"]["confidence"][low:high]),"mlp":tok.decode(row["fill_ids"]["mlp"][low:high]),"mlp_hit":a==c})
            break
assert sum(counts.values())==results["masked_tokens"]
assert all(sum(hits[g][m] for g in group_names)==n for m,n in results["hits"].items())
sizes={}
for label,name in [("width64","mlp"),("width128","wide")]:
    data=torch.load(base/f"run/policies/{name}.pt",map_location="cpu")
    sizes[label]=sum(t.numel() for t in data["state_dict"].values())
report={"checked_totals":results["hits"],"reference_token_types":counts,"correct_by_type":hits,"ranker_parameters":sizes,"examples":examples[:3]}
articles=sorted({r["article"] for r in pred})
idx=np.random.default_rng(53).integers(len(articles),size=(10000,len(articles)))
pairs=[("mlp","confidence"),("mlp","lr"),("wide","mlp"),("wide","confidence"),("wide","lr"),("mlp","prior_mlp"),("lr","prior_lr")]
report["paired_differences_pp"]={}
for treatment,comparison in pairs:
    article_totals=np.array([[sum(r["correct"][treatment]-r["correct"][comparison] for r in pred if r["article"]==a),
                              sum(r["gap_length"] for r in pred if r["article"]==a)] for a in articles])
    boot=article_totals[idx].sum(1)
    report["paired_differences_pp"][f"{treatment} minus {comparison}"]={
        "estimate":100*(results["hits"][treatment]-results["hits"][comparison])/results["masked_tokens"],
        "ci95":np.quantile(100*boot[:,0]/boot[:,1],[.025,.975]).tolist()}
(base/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
