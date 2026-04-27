#!/bin/bash

echo "════════════════════════════════════════════════════"
echo "           Flow Matching Results Comparison"
echo "════════════════════════════════════════════════════"
echo ""

for method in flow denoising reranker; do
    checkpoint="pdbbindpp_${method}_matched"
    [[ $method == "flow" ]] && checkpoint="pdbbindpp_flow_best_candidate"
    
    summary_file="checkpoints/$checkpoint/claim_summary.json"
    
    if [[ -f "$summary_file" ]]; then
        echo "【$method】"
        jq '.test | {
            method: .primary_objective,
            valid: .candidate_valid_fraction,
            pocket_fit: .strict_pocket_fit_score,
            topology_spec: .topology_specialization_score,
            geometry_spec: .geometry_specialization_score,
            pocket_spec: .pocket_specialization_score
        }' "$summary_file" 2>/dev/null | sed 's/^/  /'
        echo ""
    else
        echo "❌ 未找到结果: $summary_file"
        echo ""
    fi
done

echo "════════════════════════════════════════════════════"
echo "Key Metrics Explanation:"
echo "  - valid: Chemistry validity (1.0 = perfect)"
echo "  - pocket_fit: Strict pocket fit score (0-1)"
echo "  - *_spec: Modality specialization (0-1)"
echo "════════════════════════════════════════════════════"
