#!/usr/bin/env bash
# Steps 3-7 on a sample of users (run inside Ubuntu/WSL, after scripts/run_phase2.sh):
#     scripts/run_pipeline.sh                       # detached: survives VS Code / terminal disconnects
#     scripts/run_pipeline.sh --fg                  # foreground
#     SAMPLE_PCT=8 TT_EPOCHS=5 RANK_EPOCHS=2 scripts/run_pipeline.sh
# Follow progress:  tail -f ~/recsys_logs/latest.log
set -euo pipefail
cd "$(dirname "$0")/.."     # project root

LOG_DIR="$HOME/recsys_logs"
if [[ "${1:-}" != "--fg" ]]; then
    mkdir -p "$LOG_DIR"
    log="$LOG_DIR/pipeline_$(date +%Y%m%d_%H%M%S).log"
    ln -sf "$log" "$LOG_DIR/latest.log"
    systemd-run --user --unit="recsys-$(date +%H%M%S)" --working-directory="$PWD" \
        --setenv=SAMPLE_PCT="${SAMPLE_PCT:-5}" ${TT_EPOCHS:+--setenv=TT_EPOCHS="$TT_EPOCHS"} ${RANK_EPOCHS:+--setenv=RANK_EPOCHS="$RANK_EPOCHS"} \
        bash -c "exec '$PWD/scripts/run_pipeline.sh' --fg > '$log' 2>&1"
    echo "Pipeline started in background. Log: $log"
    echo "Follow with:  tail -f $LOG_DIR/latest.log      Stop with:  systemctl --user stop 'recsys-*'"
    exit 0
fi

source "$HOME/venvs/recsys/bin/activate"
export PYTHONPATH="$PWD" SPARK_LOCAL_IP=127.0.0.1 TF_CPP_MIN_LOG_LEVEL=2

step() { echo; echo "=== $(date +%T)  $1"; shift; "$@"; }
step "sample ${SAMPLE_PCT:-5}% of users"        python -m src.training.sample
step "step 3: two-tower retrieval"             python -m src.retrieval.two_tower
step "step 4: FAISS index"                     python -m src.index.build_faiss
step "step 5: DCN + Wide&Deep rankers"         python -m src.ranking.train_ranker
step "steps 3-6: offline evaluation"           python -m src.evaluation.evaluate
step "step 7: A/B test (test-period replay)"   python -m src.evaluation.ab_test
step "step 7: export models to Triton repo"    python -m src.serving.export_triton

# small JSON results are kept in git (models/ is not)
mkdir -p reports
cp models/*.json reports/
cp models/faiss/stats.json reports/faiss_stats.json
cp models/two_tower/train_history.json reports/two_tower_history.json
echo; echo "=== $(date +%T)  PIPELINE DONE  (reports in reports/*.json)"
