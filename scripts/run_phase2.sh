#!/usr/bin/env bash
# Step 2 launcher: Spark feature tables (run inside Ubuntu/WSL):
#     scripts/run_phase2.sh                  # detached: keeps running if VS Code / the terminal disconnects
#     scripts/run_phase2.sh --fg             # foreground
#     SAMPLE_FRAC=0.02 scripts/run_phase2.sh # quick test on ~2% of users
# Follow progress:  tail -f ~/phase2_logs/latest.log
set -euo pipefail
cd "$(dirname "$0")/.."     # project root

LOG_DIR="$HOME/phase2_logs"
if [[ "${1:-}" != "--fg" ]]; then
    mkdir -p "$LOG_DIR"
    log="$LOG_DIR/phase2_$(date +%Y%m%d_%H%M%S).log"
    ln -sf "$log" "$LOG_DIR/latest.log"
    # a systemd user unit lives outside the terminal session; WSL kills plain nohup/setsid jobs when the session closes
    systemd-run --user --unit="phase2-$(date +%H%M%S)" --working-directory="$PWD" \
        --setenv=SAMPLE_FRAC="${SAMPLE_FRAC:-}" \
        bash -c "exec '$PWD/scripts/run_phase2.sh' --fg > '$log' 2>&1"
    echo "Phase 2 started in background. Log: $log"
    echo "Follow with:  tail -f $LOG_DIR/latest.log      Stop with:  systemctl --user stop 'phase2-*'"
    exit 0
fi

source "$HOME/venvs/recsys/bin/activate"
export SPARK_LOCAL_IP=127.0.0.1   # WSL's eth0 IP changes on every restart; local mode only needs loopback
export PYTHONPATH="$PWD"          # so "from src.features import ..." resolves

# Spark scratch (shuffle/spill) on the Linux disk. /tmp is tmpfs = RAM, which starved WSL and dropped VS Code.
SCRATCH="$HOME/spark-tmp"
rm -rf "$SCRATCH"
mkdir -p "$SCRATCH"

# Full run writes to the Linux disk, then copies once to data/features (sample mode writes data/features_sample itself)
if [[ -z "${SAMPLE_FRAC:-}" ]]; then
    export PHASE2_OUT_DIR="$HOME/phase2_out"
fi

echo "=== $(date)  start  SAMPLE_FRAC=${SAMPLE_FRAC:-0}  OUT=${PHASE2_OUT_DIR:-data/features_sample}"
# local[3] leaves one core free so Windows/VS Code stay responsive; 4g heap fits inside the WSL memory limit
spark-submit \
    --master "local[3]" \
    --driver-memory 4g \
    --conf spark.local.dir="$SCRATCH" \
    --conf spark.driver.extraJavaOptions="-Djava.io.tmpdir=$SCRATCH" \
    --conf spark.ui.showConsoleProgress=false \
    src/features/run_phase2.py

if [[ -n "${PHASE2_OUT_DIR:-}" ]]; then
    echo "=== $(date)  copying outputs to data/features"
    for t in user_features item_features train_interactions; do
        rm -rf "data/features/$t.parquet"
        cp -r "$PHASE2_OUT_DIR/$t.parquet" data/features/
    done
fi
rm -rf "$SCRATCH"
echo "=== $(date)  PHASE 2 DONE"
