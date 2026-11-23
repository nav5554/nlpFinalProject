#!/usr/bin/env bash
set -euo pipefail
artifacts="$(cd "$(dirname "$0")" && pwd)"
repo="${NLP_REPO_DIR:-$(cd "$artifacts/../.." && pwd)}"
python="${NLP_PYTHON:-$repo/.venv/bin/python}"
run_dir="${1:-$artifacts/run}"
mkdir "$run_dir"
run_dir="$(cd "$run_dir" && pwd)"
ln -s "$repo/mdlm-local" "$run_dir/mdlm-local"
cd "$run_dir"
export PYTHONPATH="$repo"
{
    git -C "$repo" rev-parse HEAD
    git -C "$repo" status --short
    "$python" -V
    "$python" -c 'from importlib.metadata import version; print({p: version(p) for p in ("torch", "transformers", "datasets", "huggingface_hub", "pandas", "pyarrow", "scikit-learn", "numpy")})'
    shasum -a 256 "$repo/mdlm-local/model.safetensors" "$artifacts/pilot.py"
} | tee provenance.log
"$python" -u "$artifacts/pilot.py" collect 2>&1 | tee collect.log
"$python" -u "$repo/auroc_check.py" 2>&1 | tee auroc.log
"$python" -u "$repo/train_policies.py" 2>&1 | tee train.log
"$python" -u "$artifacts/pilot.py" evaluate 2>&1 | tee evaluate.log
