set -eu

source_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
release_root=$source_root
python=.envs/main/bin/python
while [ ! -x "$release_root/$python" ]; do
    if [ "$release_root" = / ]; then
        echo 'Create .envs/main in the repository root as described in evaluation/README.md.' >&2
        exit 1
    fi
    release_root=$(dirname -- "$release_root")
done

export PYTHONPATH="$source_root"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1

new_run=
dataset=
config=
command=${1:-}
if [ "$command" = prepare ] || [ "$command" = evaluate ]; then
    [ "$command" != prepare ] || new_run=1
    for argument in "$@"; do
        case "$argument" in
            --videos|--videos=*|--predictions|--predictions=*|--use-reference-videos) new_run=1 ;;
            --dataset|--dataset=*) dataset=1 ;;
            --config|--config=*) config=1 ;;
        esac
    done
fi
if [ -n "$new_run" ]; then
    [ -n "$dataset" ] || set -- "$@" --dataset "$release_root/HandWorldBench-data"
    if [ -z "$config" ]; then
        for directory in tmp xdg huggingface torch triton torchinductor torch_extensions cuda numba matplotlib; do
            mkdir -p "$release_root/runtime_cache/$directory"
        done
        set -- "$@" --config "$source_root/configs/runtime.json"
    fi
fi
exec "$release_root/$python" -m handworldbench "$@"
