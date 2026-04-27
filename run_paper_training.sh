#!/bin/bash

################################################################################
#  Flow Matching - Paper-Quality Training (扩展版本)
#  
#  这个脚本用于生成论文质量的实验结果
#  相比快速验证版本:
#    - max_steps: 8 → 100 (更长的训练)
#    - max_examples: 512 → 1024 (更大的数据集)
#    - batch_size: 8 → 32 (更大的批)
#
#  预期运行时间: 10-20 分钟/模型 (CPU)
################################################################################

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# 创建日志目录
mkdir -p training_logs_paper

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "                Flow Matching - Paper-Quality Training"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

cd /home/hugo/codes/patent_rw

# 配置对比
echo "配置对比:"
echo ""
echo "  版本              max_steps  max_examples  batch_size  预期时间"
echo "  ─────────────────────────────────────────────────────────────"
echo "  快速验证 (现有)      8          512            8         6-8s"
echo "  论文质量 (新)       100        1024           32        10-20m"
echo ""

read -p "继续运行论文版本吗? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    log_error "已取消"
    exit 1
fi

echo ""
log_info "════════════════════════════════════════════════════════════════════════════"
log_info "开始: Flow Matching (Paper Quality) - 100 steps, 1024 examples"
log_info "════════════════════════════════════════════════════════════════════════════"

start_time=$(date +%s)

# 验证配置
log_info "验证配置..."
if ! cargo run --quiet --bin pocket_diffusion -- validate --kind experiment \
    --config configs/unseen_pocket_pdbbindpp_flow_best_candidate_paper.json; then
    log_error "配置验证失败"
    exit 1
fi
log_success "配置验证通过"

# 运行训练
log_info ""
log_info "运行训练... (这需要 10-20 分钟)"
log_info "可以在另一个终端用以下命令查看进度:"
log_info "  tail -f training_logs_paper/run.log"
log_info ""

if ! cargo run --release --bin pocket_diffusion -- research experiment \
    --config configs/unseen_pocket_pdbbindpp_flow_best_candidate_paper.json \
    | tee training_logs_paper/run.log; then
    log_error "训练失败"
    exit 1
fi

end_time=$(date +%s)
duration=$((end_time - start_time))

log_success "训练完成! (耗时: $((duration / 60))m $((duration % 60))s)"

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "结果已保存到: checkpoints/pdbbindpp_flow_best_candidate_paper/"
echo ""
echo "查看结果:"
echo "  cat checkpoints/pdbbindpp_flow_best_candidate_paper/claim_summary.json | jq '.test'"
echo ""
echo "对比快速版本 vs 论文版本:"
echo "  echo '快速版本:' && cat checkpoints/pdbbindpp_flow_best_candidate/claim_summary.json | jq '.test | {valid: .candidate_valid_fraction, pocket_fit: .strict_pocket_fit_score}'"
echo "  echo '论文版本:' && cat checkpoints/pdbbindpp_flow_best_candidate_paper/claim_summary.json | jq '.test | {valid: .candidate_valid_fraction, pocket_fit: .strict_pocket_fit_score}'"
echo "════════════════════════════════════════════════════════════════════════════"

