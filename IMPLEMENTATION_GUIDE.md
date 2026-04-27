# 三维四语义解耦修复 - 实施指南

## 快速导航

### 📋 已完成的工作 (3/23)

```
✓ P1T1: src/losses/mutual_information.rs 已创建
✓ P1T2: src/losses/mod.rs 已更新 (导出MI模块)
✓ P1T3: src/training/metrics/losses.rs 已更新 (添加MI字段)
```

### ⏭️ 下一步 (Priority 1)

需要在以下文件中修改，使MI监测工作起来：

#### 1. src/training/trainer.rs (P2T1-P2T5)

**修改 1.1: 导入MutualInformationMonitor (第12行附近)**
```rust
use crate::{
    config::{AffinityWeighting, ResearchConfig},
    data::{ExampleBatchIter, MolecularExample},
    losses::{
        build_primary_objective, compute_primary_objective_batch, ConsistencyLoss, GateLoss,
        IntraRedundancyLoss, LeakageLoss, PocketGeometryAuxLoss, ProbeLoss,
        MutualInformationMonitor,  // ← 添加这行
    },
    // ...
};
```

**修改 1.2: 添加mi_monitor字段到ResearchTrainer (第35行附近)**
```rust
pub struct ResearchTrainer {
    optimizer: nn::Optimizer,
    scheduler: StageScheduler,
    checkpoints: CheckpointManager,
    primary_objective: Box<dyn TaskDrivenObjective<ResearchForward>>,
    redundancy_loss: IntraRedundancyLoss,
    probe_loss: ProbeLoss,
    leakage_loss: LeakageLoss,
    gate_loss: GateLoss,
    consistency_loss: ConsistencyLoss,
    pocket_geometry_loss: PocketGeometryAuxLoss,
    mi_monitor: MutualInformationMonitor,  // ← 添加这行
    // ... 其他字段
}
```

**修改 1.3: 在new()中初始化 (第70行附近)**
```rust
Ok(Self {
    optimizer,
    scheduler,
    checkpoints,
    primary_objective: build_primary_objective(&config.training),
    redundancy_loss: IntraRedundancyLoss::default(),
    probe_loss: ProbeLoss,
    leakage_loss: LeakageLoss::default(),
    gate_loss: GateLoss,
    consistency_loss: ConsistencyLoss::default(),
    pocket_geometry_loss: PocketGeometryAuxLoss::default(),
    mi_monitor: MutualInformationMonitor::default(),  // ← 添加这行
    // ... 其他字段
})
```

**修改 1.4: 在fit()中计算MI值 (第150行附近)**
```rust
// 在所有loss计算之后，添加MI计算
let (mi_topo_geo, mi_topo_pocket, mi_geo_pocket) = if forwards.len() > 0 {
    let (m1, m2, m3) = self.mi_monitor.compute_all_mi(&forwards[0]);
    (m1, m2, m3)
} else {
    (0.0, 0.0, 0.0)
};
```

**修改 1.5: 更新AuxiliaryLossMetrics初始化 (第200行附近)**
```rust
auxiliaries: AuxiliaryLossMetrics {
    intra_red: intra_red_value,
    probe: probe_value,
    leak: leak_value,
    gate: gate_value,
    slot: slot_value,
    consistency: consistency_value,
    pocket_contact: pocket_contact_value,
    pocket_clash: pocket_clash_value,
    mi_topo_geo,           // ← 添加这行
    mi_topo_pocket,        // ← 添加这行
    mi_geo_pocket,         // ← 添加这行
},
```

**修改 1.6: 更新日志输出 (第230行附近)**

找到这样的日志行：
```rust
println!(
    "[{}] Step {}: task={:.6} intra_red={:.6} probe={:.6} ...",
    now, self.step, primary_value, intra_red_value, probe_value
);
```

改为：
```rust
println!(
    "[{}] Step {}: task={:.6} intra_red={:.6} probe={:.6} MI_tg={:.4} MI_tp={:.4} MI_gp={:.4} total={:.6}",
    now, self.step, primary_value, intra_red_value, probe_value,
    mi_topo_geo, mi_topo_pocket, mi_geo_pocket, total_value
);
```

---

### ✅ 验证完成度

完成上述修改后，运行：

```bash
# 1. 检查编译
cargo build --release 2>&1 | head -50

# 2. 运行快速测试 (max_steps=5, max_examples=64)
cargo run --release --bin pocket_diffusion -- research experiment \
  --config configs/unseen_pocket_pdbbindpp_flow_best_candidate_paper.json 2>&1 | tail -20

# 3. 验证MI值在日志中出现
grep "MI_" training_logs_paper/run.log
```

---

### 📝 Phase 4: Stage配置修改 (可选，需要更多改动)

如果Phase 2工作正常，可以继续：

1. **src/config/types/training.rs** - 添加stage0_steps
2. **src/training/scheduler.rs** - 处理新的Stage 0
3. **configs/*.json** - 更新stage配置

详见 `todo.json` 中的 Phase 4 任务。

---

### 📊 预期结果

修复完成后的日志样例：

```
[2026-04-27 15:00:00] Step 0: task=3.426532 intra_red=93.507 probe=109.227 MI_tg=0.8234 MI_tp=0.7891 MI_gp=0.8156 total=215.456
[2026-04-27 15:00:02] Step 1: task=2.834123 intra_red=85.234 probe=98.456 MI_tg=0.7891 MI_tp=0.7234 MI_gp=0.7856 total=198.234
...
[2026-04-27 15:00:45] Step 99: task=0.011160 intra_red=1.072880 probe=7.301984 MI_tg=0.1234 MI_tp=0.0987 MI_gp=0.1456 total=1.152618
```

注意：MI值逐步下降（高度解耦的表现）

---

### 🐛 常见问题

**Q: 编译出错"MutualInformationMonitor not found"**
A: 确保修改了src/losses/mod.rs的导出

**Q: MI值为NaN或Inf**
A: 检查compute_mi()中的数值稳定性，可能需要添加epsilon值

**Q: MI值始终很小（例如0.001）**
A: 可能是正常的！取决于数据分布。关键是看它是否逐步下降

**Q: 训练变慢了**
A: MI计算增加了计算量。可以优化binning方法或降低num_bins

---

### 📚 参考文档

- `todo.json` - 完整的23个任务清单
- `/fix_plan.md` - 高层次的修复计划
- `src/losses/mutual_information.rs` - MI计算器的具体实现

---

### 🎯 下一个里程碑

完成Phase 2后，将能够：

✓ 在训练日志中看到MI值
✓ 在training_history中记录MI指标
✓ 定量验证解耦过程

然后可以决定是否继续Phase 3-6来进一步改进。

