#!/bin/bash
set -euo pipefail
repo_dir="$(pwd)"
folder="$repo_dir/data/milestone1/matched"
py="$repo_dir/.venv/bin/python"
export PYTHONPATH="$repo_dir:$repo_dir/data/milestone1/sentence/deps"
export NLTK_DATA="$repo_dir/data/milestone1/sentence/nltk_data"
export HF_HUB_OFFLINE=1
mkdir "$folder/run"
mkdir "$folder/run/data"
ln -s "$repo_dir/mdlm-local" "$folder/run/mdlm-local"
cd "$folder/run"
git -C "$repo_dir" rev-parse HEAD > provenance.txt
shasum -a 256 "$folder/experiment.py" "$repo_dir/train_policies.py" "$repo_dir/mdlm-local/model.safetensors" >> provenance.txt
"$py" -u "$folder/experiment.py" preview
"$py" -u "$folder/experiment.py" collect
"$py" -u "$repo_dir/train_policies.py" | tee train64.log
# All training settings besides width are identical. Keep the primary checkpoints.
mkdir -p wide
ln -s "$folder/run/data" wide/data
sed 's/^HIDDEN = 64$/HIDDEN = 128/' "$repo_dir/train_policies.py" > wide/train128.py
(cd wide && "$py" -u train128.py) | tee train128.log
cp wide/policies/mlp.pt policies/wide.pt
"$py" -u "$folder/experiment.py" evaluate
