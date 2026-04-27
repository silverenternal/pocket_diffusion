#!/bin/bash

################################################################################
#  Flow Matching Training Experiments (F2.2 & F4.2)
#  长时间训练脚本 - 自动保存结果
#
#  用法:
#    bash run_training_experiments.sh           # 运行三个主要方法
#    bash run_training_experiments.sh --lp      # 额外运行 LP-PDBBind 大表面
#    bash run_training_experiments.sh --help    # 显示帮助
################################################################################

set -e  # 任何错误都退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志文件
LOG_DIR="training_logs"
LOG_FILE="${LOG_DIR}/training_$(date +%Y%m%d_%H%M%S).log"
SUMMARY_FILE="${LOG_DIR}/summary.txt"

# 创建日志目录
mkdir -p "${LOG_DIR}"

# 日志函数
log_info() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "${LOG_FILE}"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1" | tee -a "${LOG_FILE}"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1" | tee -a "${LOG_FILE}"
}

log_warning() {
    echo -e "${YELLOW}[!]${NC} $1" | tee -a "${LOG_FILE}"
}

# 帮助信息
show_help() {
    cat << 'HELP_EOF'
Flow Matching Training Experiments Script
==========================================

Usage:
  bash run_training_experiments.sh [OPTIONS]

Options:
  --lp              额外运行 LP-PDBBind 大表面训练 (F4.2 补充)
  --flow-only       只运行流匹配 (F2.2), 跳过基线
  --baseline-only   只运行基线 (F4.2), 跳过流匹配
  --help            显示此帮助信息

Examples:
  # 运行三个主要方法 (推荐)
  bash run_training_experiments.sh

  # 加上 LP-PDBBind 大表面
  bash run_training_experiments.sh --lp

  # 只测试流匹配
  bash run_training_experiments.sh --flow-only

Output:
  - Training logs: training_logs/training_YYYYMMDD_HHMMSS.log
  - Results saved to:
    * checkpoints/pdbbindpp_flow_best_candidate/claim_summary.json
    * checkpoints/pdbbindpp_denoising_matched/claim_summary.json
    * checkpoints/pdbbindpp_reranker_matched/claim_summary.json
  - Summary: training_logs/summary.txt

HELP_EOF
    exit 0
}

# 解析参数
RUN_LP=0
FLOW_ONLY=0
BASELINE_ONLY=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --lp)
            RUN_LP=1
            log_info "将额外运行 LP-PDBBind 大表面"
            shift
            ;;
        --flow-only)
            FLOW_ONLY=1
            log_info "只运行流匹配 (F2.2)"
            shift
            ;;
        --baseline-only)
            BASELINE_ONLY=1
            log_info "只运行基线 (F4.2)"
            shift
            ;;
        --help)
            show_help
            ;;
        *)
            log_error "未知选项: $1"
            show_help
            ;;
    esac
done

# 检查必要的文件和工具
check_prerequisites() {
    log_info "检查前置条件..."
    
    cd /home/hugo/codes/patent_rw || {
        log_error "无法进入项目目录"
        exit 1
    }
    
    # 检查配置文件
    local configs=(
        "configs/unseen_pocket_pdbbindpp_flow_best_candidate.json"
        "configs/unseen_pocket_pdbbindpp_denoising_matched.json"
        "configs/unseen_pocket_pdbbindpp_reranker_matched.json"
    )
    
    for cfg in "${configs[@]}"; do
        if [[ ! -f "$cfg" ]]; then
            log_error "缺少配置文件: $cfg"
            exit 1
        fi
    done
    
    log_success "所有配置文件检查通过"
    
    # 检查 cargo
    if ! command -v cargo &> /dev/null; then
        log_error "cargo 未找到，请安装 Rust"
        exit 1
    fi
    
    log_success "Cargo 检查通过"
}

# 验证配置
validate_config() {
    local config=$1
    log_info "验证配置: $config"
    
    if ! cargo run --quiet --bin pocket_diffusion -- validate --kind experiment --config "$config" >> "${LOG_FILE}" 2>&1; then
        log_error "配置验证失败: $config"
        return 1
    fi
    
    log_success "配置验证通过: $config"
    return 0
}

# 运行训练
run_training() {
    local name=$1
    local config=$2
    local checkpoint=$3
    
    log_info "════════════════════════════════════════════════════"
    log_info "开始: $name"
    log_info "配置: $config"
    log_info "结果保存位置: checkpoints/$checkpoint/claim_summary.json"
    log_info "════════════════════════════════════════════════════"
    
    local start_time=$(date +%s)
    
    # 验证配置
    if ! validate_config "$config"; then
        log_error "$name 配置验证失败"
        return 1
    fi
    
    # 运行训练
    if ! cargo run --release --bin pocket_diffusion -- research experiment \
        --config "$config" >> "${LOG_FILE}" 2>&1; then
        log_error "$name 运行失败"
        return 1
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log_success "$name 完成 (耗时: $((duration / 60))m $((duration % 60))s)"
    
    # 检查结果是否保存
    if [[ -f "checkpoints/$checkpoint/claim_summary.json" ]]; then
        log_success "结果已保存: checkpoints/$checkpoint/claim_summary.json"
        return 0
    else
        log_warning "警告: 未找到结果文件"
        return 1
    fi
}

# 对比结果
compare_results() {
    log_info "════════════════════════════════════════════════════"
    log_info "结果对比"
    log_info "════════════════════════════════════════════════════"
    
    local methods=(
        "flow:pdbbindpp_flow_best_candidate"
        "denoising:pdbbindpp_denoising_matched"
        "reranker:pdbbindpp_reranker_matched"
    )
    
    local results_found=0
    
    for method_info in "${methods[@]}"; do
        IFS=':' read -r name checkpoint <<< "$method_info"
        local summary_file="checkpoints/$checkpoint/claim_summary.json"
        
        if [[ -f "$summary_file" ]]; then
            results_found=$((results_found + 1))
            log_info ""
            log_info "【$name】"
            
            # 提取关键指标
            if command -v jq &> /dev/null; then
                jq '.surfaces[0] | {
                    valid: .candidate_valid_fraction,
                    pocket_fit: .strict_pocket_fit_score
                }' "$summary_file" 2>/dev/null | sed 's/^/  /' || {
                    log_warning "jq 解析失败，显示原始内容:"
                    head -n 30 "$summary_file" | sed 's/^/  /'
                }
            else
                log_warning "jq 未安装，无法解析 JSON"
                head -n 30 "$summary_file" | sed 's/^/  /'
            fi
        else
            log_warning "未找到结果: $summary_file"
        fi
    done
    
    echo ""
    log_info "找到 $results_found 个结果"
}

# 生成总结
generate_summary() {
    log_info "════════════════════════════════════════════════════"
    log_info "生成总结报告"
    log_info "════════════════════════════════════════════════════"
    
    cat > "${SUMMARY_FILE}" << EOF
Flow Matching Training Experiments Summary
==========================================
Generated: $(date)
Log file: ${LOG_FILE}

Commands Executed:
EOF
    
    if [[ $FLOW_ONLY -eq 0 ]] || [[ $BASELINE_ONLY -eq 0 ]]; then
        echo "  1. F2.2 Flow Best Candidate" >> "${SUMMARY_FILE}"
        echo "     cargo run --release --bin pocket_diffusion -- research experiment \\" >> "${SUMMARY_FILE}"
        echo "       --config configs/unseen_pocket_pdbbindpp_flow_best_candidate.json" >> "${SUMMARY_FILE}"
        echo "" >> "${SUMMARY_FILE}"
    fi
    
    if [[ $FLOW_ONLY -eq 0 ]]; then
        echo "  2. F4.2 Denoising Baseline (matched_steps=50)" >> "${SUMMARY_FILE}"
        echo "     cargo run --release --bin pocket_diffusion -- research experiment \\" >> "${SUMMARY_FILE}"
        echo "       --config configs/unseen_pocket_pdbbindpp_denoising_matched.json" >> "${SUMMARY_FILE}"
        echo "" >> "${SUMMARY_FILE}"
        
        echo "  3. F4.2 Reranker Baseline (matched_steps=50)" >> "${SUMMARY_FILE}"
        echo "     cargo run --release --bin pocket_diffusion -- research experiment \\" >> "${SUMMARY_FILE}"
        echo "       --config configs/unseen_pocket_pdbbindpp_reranker_matched.json" >> "${SUMMARY_FILE}"
        echo "" >> "${SUMMARY_FILE}"
    fi
    
    if [[ $RUN_LP -eq 1 ]]; then
        echo "  4. F4.2 Supplement: LP-PDBBind Large Surface" >> "${SUMMARY_FILE}"
        echo "     cargo run --release --bin pocket_diffusion -- research experiment \\" >> "${SUMMARY_FILE}"
        echo "       --config configs/unseen_pocket_lp_pdbbind_refined_flow_best_candidate.json" >> "${SUMMARY_FILE}"
        echo "" >> "${SUMMARY_FILE}"
    fi
    
    cat >> "${SUMMARY_FILE}" << EOF

Results Saved To:
  - Flow Best Candidate: checkpoints/pdbbindpp_flow_best_candidate/claim_summary.json
  - Denoising Baseline:  checkpoints/pdbbindpp_denoising_matched/claim_summary.json
  - Reranker Baseline:   checkpoints/pdbbindpp_reranker_matched/claim_summary.json
EOF
    
    if [[ $RUN_LP -eq 1 ]]; then
        echo "  - LP-PDBBind Flow:     checkpoints/lp_pdbbind_refined_flow_best_candidate/claim_summary.json" >> "${SUMMARY_FILE}"
    fi
    
    cat >> "${SUMMARY_FILE}" << EOF

Comparison Command:
  for method in flow denoising reranker; do
    echo "=== \$method ==="
    cat "checkpoints/pdbbindpp_\${method}_matched/claim_summary.json" 2>/dev/null | \\
      jq '.surfaces[0] | {valid: .candidate_valid_fraction, pocket_fit: .strict_pocket_fit_score}' || echo "Not complete"
  done

Full Log: ${LOG_FILE}
EOF
    
    log_success "总结报告已保存: ${SUMMARY_FILE}"
}

################################################################################
#                              Main Execution
################################################################################

main() {
    log_info "╔════════════════════════════════════════════════════╗"
    log_info "║  Flow Matching Training Experiments (F2.2 & F4.2)  ║"
    log_info "║  启动时间: $(date '+%Y-%m-%d %H:%M:%S')                   ║"
    log_info "╚════════════════════════════════════════════════════╝"
    log_info ""
    
    # 检查前置条件
    check_prerequisites
    log_info ""
    
    # 运行实验
    declare -a failed_experiments
    
    # F2.2: Flow Best Candidate
    if [[ $BASELINE_ONLY -eq 0 ]]; then
        if ! run_training \
            "F2.2: Flow Matching Best Candidate (PDBBind++)" \
            "configs/unseen_pocket_pdbbindpp_flow_best_candidate.json" \
            "pdbbindpp_flow_best_candidate"; then
            failed_experiments+=("F2.2 Flow Best Candidate")
        fi
        log_info ""
    fi
    
    # F4.2: Denoising Baseline
    if [[ $FLOW_ONLY -eq 0 ]]; then
        if ! run_training \
            "F4.2 Baseline 1: Conditioned Denoising (matched_steps=50)" \
            "configs/unseen_pocket_pdbbindpp_denoising_matched.json" \
            "pdbbindpp_denoising_matched"; then
            failed_experiments+=("F4.2 Denoising Baseline")
        fi
        log_info ""
    fi
    
    # F4.2: Reranker Baseline
    if [[ $FLOW_ONLY -eq 0 ]]; then
        if ! run_training \
            "F4.2 Baseline 2: Calibrated Reranker (matched_steps=50)" \
            "configs/unseen_pocket_pdbbindpp_reranker_matched.json" \
            "pdbbindpp_reranker_matched"; then
            failed_experiments+=("F4.2 Reranker Baseline")
        fi
        log_info ""
    fi
    
    # F4.2 Supplement: LP-PDBBind (Optional)
    if [[ $RUN_LP -eq 1 ]]; then
        if ! run_training \
            "F4.2 Supplement: LP-PDBBind Flow Best Candidate (Large Surface)" \
            "configs/unseen_pocket_lp_pdbbind_refined_flow_best_candidate.json" \
            "lp_pdbbind_refined_flow_best_candidate"; then
            failed_experiments+=("F4.2 LP-PDBBind Flow")
        fi
        log_info ""
    fi
    
    # 对比结果
    compare_results
    log_info ""
    
    # 生成总结
    generate_summary
    log_info ""
    
    # 最终状态
    log_info "════════════════════════════════════════════════════"
    if [[ ${#failed_experiments[@]} -eq 0 ]]; then
        log_success "所有实验完成！"
        log_info ""
        log_info "结果已保存到:"
        log_info "  - checkpoints/pdbbindpp_flow_best_candidate/"
        log_info "  - checkpoints/pdbbindpp_denoising_matched/"
        log_info "  - checkpoints/pdbbindpp_reranker_matched/"
        [[ $RUN_LP -eq 1 ]] && log_info "  - checkpoints/lp_pdbbind_refined_flow_best_candidate/"
        log_info ""
        log_info "详见: ${LOG_FILE}"
        return 0
    else
        log_error "以下实验失败:"
        for exp in "${failed_experiments[@]}"; do
            log_error "  - $exp"
        done
        log_info ""
        log_info "详见: ${LOG_FILE}"
        return 1
    fi
}

# 捕获异常退出
trap 'log_error "脚本被中断"; exit 1' SIGINT SIGTERM

# 执行
main
exit $?
