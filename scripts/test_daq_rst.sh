set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_PATH="$ROOT_DIR/projects/configs/sparsedrive_small_stage2_daq_rst.py"
CHECKPOINT_PATH="$ROOT_DIR/checkpoints/sparsedrive_stage2.pth"
ANN_FILE="$ROOT_DIR/data/infos/daq_data_infos_infe.pkl"
RESULT_PATH="$ROOT_DIR/work_dirs/sparsedrive_small_stage2_daq_rst/results.pkl"

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Missing config file: $CONFIG_PATH"
    exit 1
fi

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
    echo "Missing checkpoint file: $CHECKPOINT_PATH"
    exit 1
fi

if [[ ! -f "$ANN_FILE" ]]; then
    echo "Missing annotation file: $ANN_FILE"
    echo "Run the DAQ converter first to generate it."
    exit 1
fi

export PYTHONPATH="$ROOT_DIR":${PYTHONPATH:-}
bash "$ROOT_DIR/tools/dist_test.sh" \
    "$CONFIG_PATH" \
    "$CHECKPOINT_PATH" \
    1 \
    --deterministic \
    --format-only \
    --out "$RESULT_PATH" \
    --cfg-options data.test.ann_file="$ANN_FILE" data.val.ann_file="$ANN_FILE" eval_config.ann_file="$ANN_FILE"
