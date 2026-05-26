set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_PATH="$ROOT_DIR/projects/configs/sparsedrive_small_stage2_daq_rst.py"
RESULT_PATH="$ROOT_DIR/work_dirs/sparsedrive_small_stage2_daq_rst/results.pkl"
SCORE_THRESH="${1:-0.3}"
MAP_SCORE_THRESH="${2:-0.3}"

if [[ ! -f "$RESULT_PATH" ]]; then
    echo "Missing result file: $RESULT_PATH"
    echo "Run scripts/test_daq_rst.sh first to generate it."
    exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Missing config file: $CONFIG_PATH"
    exit 1
fi

export PYTHONPATH="$ROOT_DIR":$PYTHONPATH
python "$ROOT_DIR/tools/visualization/visualize.py" \
	"$CONFIG_PATH" \
    --result-path "$RESULT_PATH" \
    --score-thresh "$SCORE_THRESH" \
    --map-score-thresh "$MAP_SCORE_THRESH"
