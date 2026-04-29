# 数据集、解析流程与训练方法说明

本文档面向当前 `pocket_diffusion` / `patent_rw` 仓库的 Rust 研究栈，说明项目当前使用和支持的数据集格式、下载与整理方法、parser 逻辑，以及配置驱动训练流程。文档反映 2026-04-29 当前代码状态。

需要先明确当前边界：系统保留 target-ligand conditioned denoising/refinement 和 geometry-only coordinate flow 作为基线，同时已经实现 config-gated de novo full molecular flow。真正的 de novo 路径要求 `generation_mode=de_novo_initialization`、`generation_method.primary_backend.family=flow_matching`、`generation_method.flow_matching.geometry_only=false`，并启用 `geometry/atom_type/bond/topology/pocket_context` 五个 flow branch。数据集解析仍以蛋白-配体复合物为中心，原始 pocket 坐标记录 ligand-centered model frame；de novo 执行路径会对 pocket/context 重新居中，并将目标 ligand 仅作为训练监督，而不是初始化或条件输入。

## 1. 当前使用的数据集层级

### 1.1 仓库自带 mini_pdbbind

路径：

```text
examples/datasets/mini_pdbbind/
```

用途：

- 单元测试、smoke test、配置验证。
- 演示 manifest_json、PDBbind-like 目录扫描和 affinity label attachment。
- 不用于 claim-bearing 结论，因为只有 4 个 protein-ligand complex。

当前结构：

```text
examples/datasets/mini_pdbbind/
  manifest.json
  affinity_labels.csv
  INDEX_affinity_data.2020
  prot_a/protein.pdb
  prot_a/ligand.sdf
  prot_b/protein.pdb
  prot_b/ligand.sdf
  ...
```

mini 数据无需下载，clone 仓库后即可使用：

```bash
cargo run --bin pocket_diffusion -- research inspect --config configs/research_manifest.json
cargo run --bin pocket_diffusion -- research train --config configs/research_manifest.json
```

### 1.2 PDBbind-like 本地目录

路径示例：

```text
data/PDBbind_v2020_refined/
  1a30/1a30_protein.pdb
  1a30/1a30_ligand.sdf
  1bcu/1bcu_protein.pdb
  1bcu/1bcu_ligand.sdf
  ...
```

对应配置示例：

```text
configs/research_pdbbind_dir.json
```

`dataset_format = "pdbbind_like_dir"` 时，Rust discovery 会扫描 `root_dir` 的一级子目录；每个子目录里至少需要一个 `.pdb` 和一个 `.sdf`。在 `strict` 解析模式下，每个 complex 目录必须恰好有一个 `.pdb` 和一个 `.sdf`，否则 fail fast；在默认 `lightweight` 模式下，按排序后的第一个 PDB/SDF 作为该 complex 的 source。

下载与整理建议：

1. 从 PDBbind 官方渠道或可复现的公开镜像下载符合授权要求的 PDBbind v2020 refined/general 结构文件。官方数据通常需要遵守 PDBbind 的许可和引用要求。
2. 解压到 `data/PDBbind_v2020_refined/` 或自定义目录。
3. 保证每个 complex 子目录内至少包含一个 protein `.pdb` 和一个 ligand `.sdf`。
4. 如果有 affinity label 表，放到 CSV/TSV 或 PDBbind index 格式，并在 config 中设置 `data.label_table_path`。
5. 先跑 inspect：

```bash
cargo run --bin pocket_diffusion -- research inspect --config configs/research_pdbbind_dir.json
```

### 1.3 PDBbind++ 2020 manifest_json

当前仓库中已有面向 PDBbind++ 2020 refined-set layout 的整理脚本：

```text
tools/build_pdbbindpp_manifest.py
tools/build_pdbbindpp_affinity_labels.py
```

本地期望结构：

```text
data/pdbbindpp-2020/
  pbpp-2020.zip
  extracted/pbpp-2020/
    10gs/10gs_pocket.pdb
    10gs/10gs_ligand.sdf
    184l/184l_pocket.pdb
    184l/184l_ligand.sdf
    ...
  manifest.json
  affinity_labels.csv
```

当前配置示例：

```text
configs/unseen_pocket_pdbbindpp_real_backends.json
configs/unseen_pocket_pdbbindpp_flow_best_candidate_paper.json
configs/flow_matching_experiment.json
```

下载与整理示例：

```bash
python3 -m venv .venv-download
. .venv-download/bin/activate
pip install -U huggingface_hub

huggingface-cli download photonmz/pdbbindpp-2020 \
  --repo-type dataset \
  --local-dir data/pdbbindpp-2020

unzip -q data/pdbbindpp-2020/pbpp-2020.zip -d data/pdbbindpp-2020/extracted

python3 tools/build_pdbbindpp_manifest.py \
  --root data/pdbbindpp-2020/extracted/pbpp-2020 \
  --output data/pdbbindpp-2020/manifest.json
```

Affinity labels 有两种方式：

- 如果已有 `data/pdbbindpp-2020/affinity_labels.csv`，直接使用。
- 如果只有上游元数据 CSV，可用仓库脚本生成：

```bash
python3 tools/build_pdbbindpp_affinity_labels.py \
  --source-csv <upstream_affinity_metadata.csv> \
  --manifest data/pdbbindpp-2020/manifest.json \
  --output data/pdbbindpp-2020/affinity_labels.csv
```

注意：`build_pdbbindpp_affinity_labels.py` 只负责把上游 affinity record 整理为本仓库 label table，不负责下载授权受限的原始标签文件。

### 1.4 LP-PDBBind refined companion surface

当前仓库还保留了 LP-PDBBind refined companion surface 的 manifest/label 入口：

```text
data/lp_pdbbind_refined/manifest.json
data/lp_pdbbind_refined/affinity_labels.csv
configs/unseen_pocket_lp_pdbbind_refined_real_backends.json
```

该 surface 主要用于第二个较大 benchmark anchor。它复用 protein-ligand complex manifest 和标签表，但 label table 中会带有 `source_benchmark`、`benchmark_split`、`measurement_family`、`normalization_provenance` 等额外列。Rust label parser 当前只消费它认识的 key/affinity 字段；额外列保留在源文件里，作为上游数据出处和 split 语义的审计信息。

## 2. 数据格式规范

### 2.1 Research config 的数据入口

所有 config-driven 路径都使用 `DataConfig`：

```json
{
  "data": {
    "root_dir": "./examples/datasets/mini_pdbbind",
    "dataset_format": "manifest_json",
    "manifest_path": "./examples/datasets/mini_pdbbind/manifest.json",
    "label_table_path": "./examples/datasets/mini_pdbbind/affinity_labels.csv",
    "max_ligand_atoms": 64,
    "max_pocket_atoms": 256,
    "pocket_cutoff_angstrom": 6.0,
    "batch_size": 2,
    "split_seed": 42,
    "val_fraction": 0.25,
    "test_fraction": 0.25,
    "stratify_by_measurement": true
  }
}
```

支持的 `dataset_format`：

| 格式 | 用途 | 必要字段 |
| --- | --- | --- |
| `synthetic` | 内置 toy examples，主要用于测试 | 无外部文件 |
| `manifest_json` | 推荐的可复现格式，显式列出每个 complex 的 PDB/SDF 路径 | `manifest_path` |
| `pdbbind_like_dir` | 扫描每个 complex 子目录，适合 PDBbind 风格目录 | `root_dir` |

### 2.2 manifest_json

manifest 是 split-agnostic 的 complex 列表。最小格式如下：

```json
{
  "entries": [
    {
      "example_id": "prot_a_complex_0",
      "protein_id": "prot_a",
      "pocket_path": "prot_a/protein.pdb",
      "ligand_path": "prot_a/ligand.sdf"
    }
  ]
}
```

字段含义：

| 字段 | 含义 |
| --- | --- |
| `example_id` | 稳定样本 ID，用于 artifact、label match 和去重 |
| `protein_id` | unseen-pocket split 的 group key；同一 protein_id 不应跨 train/val/test |
| `pocket_path` | protein/pocket PDB 文件路径 |
| `ligand_path` | ligand SDF 文件路径 |
| `affinity_kcal_mol` | 可选，已归一化到 kcal/mol 的 ΔG |
| `affinity_measurement_type` | 可选，原始测量族，如 `Kd`、`Ki`、`IC50`、`pKd` |
| `affinity_raw_value` / `affinity_raw_unit` | 可选，原始值和单位 |
| `affinity_normalization_provenance` | 可选，归一化来源 |
| `affinity_is_approximate` | 可选，是否是近似 affinity proxy |

相对路径以 `manifest.json` 所在目录为基准解析。这一点很重要，因为 PDBbind++ 和 LP-PDBBind refined manifest 都依赖相对路径实现可移动的数据目录。

### 2.3 PDBbind-like directory

目录扫描格式如下：

```text
dataset_root/
  complex_001/*.pdb
  complex_001/*.sdf
  complex_002/*.pdb
  complex_002/*.sdf
```

解析规则：

- 每个一级子目录代表一个 complex。
- 子目录名作为 `example_id` 和 `protein_id`。
- 默认 `lightweight` 模式选择排序后的第一个 `.pdb` 和第一个 `.sdf`。
- `strict` 模式要求恰好一个 `.pdb` 和一个 `.sdf`，否则报错。
- 若配置了 `label_table_path`，label 会按 `example_id` 或 `protein_id` 贴到 entry 上。

### 2.4 ligand SDF

当前 SDF parser 支持最小 V2000 SDF：

- 必须存在包含 `V2000` 的 counts line。
- 从 counts line 读取 atom count 和 bond count。
- atom line 至少包含 `x y z element`。
- bond line 支持固定宽度 V2000 或 whitespace split。
- bond type 保留 `1..=4`，其他值归一化为 `0`。
- atom element 映射为 `C/N/O/S/H/Other`。

SDF parser 的结果会变成：

- topology atom types
- bond index pair
- bond type tensor
- adjacency matrix
- ligand coordinates
- pairwise distance matrix
- heuristic chemistry role features

### 2.5 pocket/protein PDB

PDB parser 读取 `ATOM` 和 `HETATM` 记录：

- 坐标来自 PDB 固定列 `30..38`、`38..46`、`46..54`。
- element 优先读 `76..78`；缺失时回退到 atom name 区域。
- 原子类型同样映射为 `C/N/O/S/H/Other`。
- 根据 ligand centroid 和 `pocket_cutoff_angstrom` 选择局部 pocket atom。
- 如果 cutoff 内没有 pocket atom：
  - `lightweight` 模式使用最近 64 个 protein atom 作为 fallback pocket。
  - `strict` 模式直接报错。

这个 parser 不是完整 PDB 化学 parser；它只抽取当前模型需要的坐标和粗粒度原子类型。

### 2.6 affinity label table

`label_table_path` 支持 CSV、TSV 和 PDBbind index-like 文本。

CSV/TSV 支持的列名：

| 列 | 说明 |
| --- | --- |
| `example_id` | 优先匹配 manifest entry 的 example_id |
| `protein_id` | example_id 未匹配时按 protein_id 匹配 |
| `affinity_kcal_mol` / `affinity` / `label` | 已归一化 ΔG，单位 kcal/mol |
| `measurement_type` / `affinity_type` | 测量族 |
| `raw_value` / `affinity_value` | 原始数值 |
| `raw_unit` / `affinity_unit` | 原始单位 |
| `affinity_record` / `measurement` | 紧凑记录，如 `Kd=25nM` |

支持的 measurement：

- `Kd`
- `Ki`
- `IC50`
- `EC50`
- `pKd`
- `pKi`
- `dG`

支持的浓度单位：

- `M`
- `mM`
- `uM` / `μM`
- `nM`
- `pM`
- `fM`

归一化规则：

```text
ΔG = R * T * ln(C_molar)
R = 0.0019872041 kcal/mol/K
T = 298.15 K
```

`IC50` 和 `EC50` 会被标记为 approximate，因为它们不是严格热力学 binding constant。parser 会把 warning 写入 `DatasetValidationReport.normalization_warning_messages`，同时记录 approximate label fraction。

标签匹配优先级：

1. `example_id`
2. `protein_id`

如果 label table 中有重复 key，后出现的行覆盖先出现的行；覆盖数量会记录到 validation report。未匹配的 label row 也会记录，避免静默丢失。

## 3. Rust parser 流程

入口：

```text
src/data/dataset/core.rs
InMemoryDataset::load_from_config
```

核心流程：

1. 读取 `DataConfig`，初始化 `DatasetValidationReport`。
2. 根据 `dataset_format` 选择数据来源：
   - `synthetic`：构造内置 toy examples。
   - `manifest_json`：读取 manifest，解析相对路径。
   - `pdbbind_like_dir`：扫描 `root_dir` 下的 complex 子目录。
3. 如存在 `label_table_path`，调用 label parser 并 attach 到 manifest entries。
4. 对每个 entry：
   - `load_ligand_from_sdf`
   - 计算 ligand centroid
   - `load_pocket_from_pdb`
   - 可选 rotation augmentation
   - 转换成 `MolecularExample`
5. 应用 `generation_target`，生成 decoder supervision：
   - atom mask corruption
   - coordinate noise
   - rollout target metadata
6. 应用 `quality_filters`：
   - label coverage
   - fallback fraction
   - ligand/pocket atom count
   - source structure provenance
   - affinity metadata completeness
   - approximate label fraction
   - normalization provenance coverage
   - target-ligand context leakage rejection
7. 应用 `max_examples` 截断。
8. 完成 validation report，包括 coordinate frame、target context leakage、source reconstruction support。

当前数据加载是 in-memory 路径：`InMemoryDataset` 会 materialize 一个 `Vec<MolecularExample>`。仓库已经预留 `MolecularExampleSource` trait 给未来 lazy/cached/sharded loader，但核心训练路径没有 Python dataloader。

## 4. 坐标与上下文合同

当前模型输入使用 ligand-centered model frame：

- ligand 坐标减去 ligand centroid。
- pocket 坐标同样以 ligand centroid 为原点。
- candidate artifact 中的 `candidate.coords` 也是 ligand-centered model-frame coordinates。
- `coordinate_frame_origin` 用于恢复 source-frame coordinates。

这对 target-ligand denoising/refinement 是合理的，因为训练任务明确从带噪目标配体恢复原配体。对 de novo 路径，forward 会用 pocket-conditioned scaffold 替代目标 ligand topology/geometry encoder 输入，并对 pocket features 做 pocket-centered 处理；目标 ligand atom type、bond、topology 和坐标只进入监督损失。

相关 validation 字段：

```text
coordinate_frame_contract
coordinate_frame_artifact_contract
source_coordinate_reconstruction_supported
target_ligand_context_dependency_detected
target_ligand_context_dependency_allowed
target_ligand_context_dependency_rejected
target_ligand_context_leakage_warnings
```

如果要做 pocket-only 或 de novo claim-bearing run，配置应启用：

```json
{
  "quality_filters": {
    "reject_target_ligand_context_leakage": true
  }
}
```

对严格 claim-bearing 数据集，仍建议重建 pocket-only coordinate frame 或让 quality filter 拒绝 retained target-ligand context dependency。模型执行层已经避免把目标 ligand topology/geometry 当作 de novo 条件输入。

评估层也遵守这个边界：当 de novo 初始化得到的 ligand atom count 与监督目标不同，topology adjacency、geometry distance 和 pharmacophore role probe baseline 不会强行与目标 ligand 矩阵相减或计算 BCE，而是标记为 prediction unavailable。这样可以保留 target supervision 用于训练，同时避免把 target-shaped probe comparison 误报为 de novo 条件证据。

## 5. split 与数据质量审计

训练和实验都使用 protein-level split：

```text
InMemoryDataset::split_by_protein_fraction_with_options
```

关键配置：

```json
{
  "split_seed": 42,
  "val_fraction": 0.15,
  "test_fraction": 0.15,
  "stratify_by_measurement": true
}
```

规则：

- 先按 `protein_id` 分组。
- 同一 `protein_id` 只会进入 train、val、test 中的一个 split。
- `stratify_by_measurement=true` 时，先按 dominant measurement family 建 bucket，再交错分配 group，降低 Kd/Ki/IC50 等 label family 偏斜。
- split report 会记录 protein overlap、duplicate example id、measurement family skew、atom count skew。

claim-bearing 配置可以强制 held-out diversity threshold：

```json
{
  "quality_filters": {
    "min_validation_protein_families": 10,
    "min_test_protein_families": 10,
    "min_validation_measurement_families": 2,
    "min_test_measurement_families": 2
  }
}
```

`research inspect` 会报告 split 质量；`research train` 和 `research experiment` 会在 configured threshold 不满足时 fail fast。

## 6. 训练方法

### 6.1 训练入口

主要 CLI：

```bash
# 数据检查
cargo run --bin pocket_diffusion -- research inspect --config configs/research_manifest.json

# 训练
cargo run --bin pocket_diffusion -- research train --config configs/research_manifest.json

# 从 latest checkpoint resume
cargo run --bin pocket_diffusion -- research train --config configs/research_manifest.json --resume

# unseen-pocket experiment，包括训练、验证/test、candidate layer artifact 和 claim summary
cargo run --bin pocket_diffusion -- research experiment --config configs/unseen_pocket_manifest.json

# 配置合法性检查
cargo run --bin pocket_diffusion -- validate --kind experiment --config configs/unseen_pocket_manifest.json
```

### 6.2 模型输入与分支

每个 `MolecularExample` 进入模型前包含三类输入：

| 模态 | 主要内容 | 当前来源 |
| --- | --- | --- |
| topology | atom type、bond edge、bond type、adjacency、chemistry roles | ligand SDF |
| geometry | ligand-centered coordinates、pairwise distances | ligand SDF |
| pocket/context | pocket atom coords、atom features、pooled features、chemistry roles | protein/pocket PDB |

模型保持三路 encoder 分离：

- topology encoder
- geometry encoder
- pocket/context encoder

后续通过 slot decomposition 和 gated cross-modal interaction 交互。跨模态交互是显式 gated attention，不是 unrestricted full fusion。

### 6.3 primary objective

`training.primary_objective` 支持：

| Objective | 语义 | Claim 边界 |
| --- | --- | --- |
| `surrogate_reconstruction` | encoder/slot latent reconstruction，加少量 decoder bootstrap | debug/bootstrap baseline，不是生成质量主张 |
| `conditioned_denoising` | 默认目标；对目标 ligand 做 atom mask 和 coordinate noise，然后训练 decoder 恢复 atom type/coords/pairwise/centroid/pocket anchor | target-ligand denoising/refinement，不是 de novo |
| `flow_matching` | 默认可作为 geometry-only velocity + endpoint loss；当 `geometry_only=false` 且五个 branch 全开时，训练 coordinate、atom-type、bond、topology、pocket/context full molecular flow | geometry-only 配置只能声称坐标流；full branch 配置可作为 de novo molecular flow |
| `denoising_flow_matching` | denoising 和 flow matching 的 hybrid objective | 混合训练目标，不是新的 generation mode |

`conditioned_denoising` 的核心训练项：

- corrupted atom 的 atom type recovery
- preserved atom 的 atom type preservation
- masked coordinate denoising
- coordinate preservation
- direct noise recovery
- pairwise distance recovery
- centroid recovery
- pocket anchor loss

`rollout_eval_*` 指标只作为 detached evaluation diagnostics 记录。当前 `training.enable_trainable_rollout_loss=true` 会被配置验证拒绝，因为 tensor-preserving rollout loss 还未实现。

### 6.4 flow matching 与 full molecular flow

geometry-only 配置覆盖坐标速度场：

- velocity loss：预测从 noisy/corrupted geometry 或 deterministic scaffold 到 target geometry 的速度。
- endpoint loss：约束积分末端接近监督目标。
- `generation_method.flow_matching.geometry_only=true` 时，atom type、bond、topology 不由 flow 更新。

full molecular flow 配置要求：

- `generation_method.flow_matching.geometry_only=false`
- `multi_modal.enabled_branches=["geometry","atom_type","bond","topology","pocket_context"]`
- `multi_modal.claim_full_molecular_flow=true` 仅在上述 branch 全开时通过验证
- `multi_modal.branch_loss_weights` 和 `multi_modal.branch_schedule` 必须作为实验语义的一部分记录；zero-weight branch 可保留 rollout/artifact 产物，但其 loss graph 会跳过，不能被描述为已优化的 branch

额外训练项：

- atom-type categorical flow loss
- bond existence + bond type loss
- topology synchronization loss
- pocket/context representation consistency loss
- branch synchronization loss

de novo rollout 不再直接对 bond logits 做独立阈值化。当前 native graph
payload 通过 topology-synchronized extractor 生成：它联合 bond logits、
topology logits、坐标距离先验、连通性 pass 和保守 atom-type valence budget，
再把 `native_bonds` / `native_bond_types` 写入 raw rollout artifact。这样 raw
model-native 层能携带模型自己的图结构，同时仍与后续 repair、valence pruning、
rerank 层分开统计。

训练 checkpoint/replay metadata 会记录 generation mode、flow contract version、branch schedule hash、raw/processed evaluation contract，以及 corruption/sampling seed。resume 或 replay-check 遇到这些字段不兼容时应拒绝严格 replay，而不是只比较权重文件是否存在。

### 6.5 staged training

Stage scheduler 根据 step 激活不同辅助目标，并在每个 stage 内做线性 warmup。默认原则：

| Stage | 主要目标 |
| --- | --- |
| Stage 1 | primary objective + consistency |
| Stage 2 | 加入 intra-modality redundancy；可按权重启用 pocket/contact/clash/valence/bond-length guardrails |
| Stage 3 | 加入 semantic probes、leakage control、pharmacophore role probes/leakage |
| Stage 4 | 加入 gate sparsity 和 slot sparsity/balance |

对应权重字段：

```json
{
  "loss_weights": {
    "alpha_task": 1.0,
    "beta_intra_red": 0.1,
    "gamma_probe": 0.2,
    "delta_leak": 0.05,
    "eta_gate": 0.05,
    "mu_slot": 0.05,
    "nu_consistency": 0.1
  }
}
```

训练中会记录 objective execution plan，区分：

- active
- inactive_zero_weight
- active_zero_weighted_value
- detached diagnostics

这对于 ablation 和 claim wording 很重要，因为一个指标存在不等于它参与了 optimizer。

### 6.6 auxiliary losses

当前辅助目标包括：

- `L_consistency`：topology-geometry consistency。
- `L_intra_red`：模态内 redundancy reduction，避免 slot/latent 冗余。
- `L_probe`：语义 probe，如 geometry distance、affinity、pharmacophore role。
- `L_leak`：off-modality leakage control 或 leakage proxy。
- `L_gate`：cross-modal gate sparsity。
- `L_slot`：slot sparsity 和 balance。
- chemistry guardrails：pocket contact、pocket clash、pocket envelope、valence guardrail、bond length guardrail。

训练报告会把 optimizer-facing loss 和 detached diagnostic 分开。尤其是 leakage/probe 指标不能单独证明 representation 没有泄漏，只能作为当前协议下的风险信号。

### 6.7 affinity supervision 与 measurement family weighting

`training.affinity_weighting` 支持：

- `none`
- `inverse_frequency`

当使用混合 `Kd/Ki/IC50/pKd` 标签时，`inverse_frequency` 会按 measurement family 反频率加权，避免某一类 label family 在 probe/affinity 目标中占比过高。`IC50/EC50` 仍会被标记为 approximate，claim-bearing 运行应关注 approximate label fraction。

### 6.8 optimizer、batching、checkpoint

当前 trainer：

- 使用 `tch` / Rust 实现。
- optimizer 为 Adam。
- batch 来源是 `ExampleBatchSampler` 或 `MolecularExampleSource` adapter。
- 支持 deterministic data order：
  - `training.data_order.shuffle`
  - `training.data_order.sampler_seed`
  - `training.data_order.drop_last`
  - `training.data_order.max_epochs`
- 支持 gradient health diagnostics。
- 可选 global norm clipping。
- checkpoint 写入 `training.checkpoint_dir`。
- resume 恢复 weights、step、history 和 optimizer/scheduler metadata；但当前 tch Adam 内部 moment buffers 不做 strict serialization，所以默认不是 exact optimizer replay。
- step 执行已经拆分为 stage selection、objective execution、gradient step、runtime tracking、metric construction 和 checkpoint trigger；runtime metrics 会记录 batched forward count、per-example fallback count、forward execution mode，以及 de novo per-example fallback reason。

主要 artifact：

```text
training_summary.json
dataset_validation_report.json
split_report.json
config.snapshot.json
latest.ot / latest.json
step-N.ot / step-N.json
best.ot / best.json
run_artifacts.json
```

`research experiment` 还会写：

```text
experiment_summary.json
claim_summary.json
generation_layers_validation.json
generation_layers_test.json
candidate_metrics_validation.jsonl
candidate_metrics_test.jsonl
```

## 7. 推荐复现实验流程

### 7.1 快速 smoke

```bash
cargo run --bin pocket_diffusion -- validate --kind research --config configs/research_manifest.json
cargo run --bin pocket_diffusion -- research inspect --config configs/research_manifest.json
cargo run --bin pocket_diffusion -- research train --config configs/research_manifest.json
```

### 7.2 compact unseen-pocket experiment

```bash
cargo run --bin pocket_diffusion -- validate --kind experiment --config configs/unseen_pocket_manifest.json
cargo run --bin pocket_diffusion -- research experiment --config configs/unseen_pocket_manifest.json
```

### 7.3 PDBbind++ larger-data run

先确保数据存在：

```bash
test -f data/pdbbindpp-2020/manifest.json
test -f data/pdbbindpp-2020/affinity_labels.csv
```

再检查配置：

```bash
cargo run --bin pocket_diffusion -- validate --kind experiment --config configs/unseen_pocket_pdbbindpp_real_backends.json
cargo run --bin pocket_diffusion -- research inspect --config configs/unseen_pocket_pdbbindpp_real_backends.json
```

然后运行 experiment：

```bash
cargo run --bin pocket_diffusion -- research experiment --config configs/unseen_pocket_pdbbindpp_real_backends.json
```

如果启用 RDKit/Vina/GNINA 等外部 backend，还需要先准备对应 Python 环境和 executable。backend 结果是 score-only / heuristic / adapter evidence，不是实验 binding affinity。

### 7.4 当前最终 compact smoke 证据

2026-04-29 的 Q14 最终烟测使用三个小配置覆盖 conditioned denoising、geometry-only flow 和 de novo full-flow execution surface：

```text
configs/q14_final_smoke_conditioned_denoising.json
configs/q14_final_smoke_geometry_flow.json
configs/q14_final_smoke_de_novo_full_flow.json
```

同一目录中还保留对应 training 入口使用的 research config：

```text
configs/q14_final_smoke_conditioned_denoising.research.json
configs/q14_final_smoke_geometry_flow.research.json
configs/q14_final_smoke_de_novo_full_flow.research.json
```

对应训练、experiment、generation 和 claim 产物位于：

```text
checkpoints/q14_final_smoke/conditioned_denoising/
checkpoints/q14_final_smoke/geometry_flow/
checkpoints/q14_final_smoke/de_novo_full_flow/
```

每个目录都包含 `training_summary.json`、`experiment_summary.json`、`generation_layers_validation.json`、`generation_layers_test.json`、`claim_summary.json`、`repair_case_audit_*.json`、`frozen_leakage_probe_audit.json` 和 candidate metrics JSONL。三组烟测的 validation/test `finite_forward_fraction` 都是 1.0，记录到的 nonfinite gradient tensor 总数都是 0。

`de_novo_full_flow` 使用 `generation_mode=de_novo_initialization`、五个 full molecular flow branch 和 `target_alignment_policy=hungarian_distance`；matching provenance 记录在 `training_history[].losses.primary.branch_schedule.entries[]`。这只证明 de novo full-flow 执行路径、非索引 matching、leakage audit、raw-native summary 和 generation layer artifact 可以一起跑通；它不是大规模 held-out-pocket benchmark，也不能单独支持 de novo 生成质量结论。

raw-native 层与 repaired/inferred-bond/reranked 等 processed 层分开记录。任何 repaired 层改进都必须引用 `postprocessing_repair_audit` 或 `generation_layers_*.json.repair_case_audit`，不能描述成 raw generation evidence。完整摘要见 `docs/q14_final_smoke_summary.md`。

## 8. 当前实现限制

1. 数据加载仍是 in-memory，不适合无限扩展到超大规模 shard；未来应实现 lazy `MolecularExampleSource`。
2. PDB/SDF parser 是任务定制 parser，不是完整化学标准库替代品。
3. 数据源 pocket 坐标仍记录 ligand-centered provenance；de novo 执行路径会 pocket-center，但严格 claim-bearing 数据集仍应开启 target-context leakage rejection。
4. `flow_matching` 默认是 geometry-only；full molecular flow 需要显式关闭 `geometry_only` 并启用五个 branch。
5. `rollout_eval_*` 是 detached diagnostics。
6. mini_pdbbind 只能作为 smoke，不是 claim-bearing benchmark。
7. external backend score-only 指标不能被描述为实验亲和力。

## 9. 关键代码路径

| 主题 | 路径 |
| --- | --- |
| DataConfig | `src/config/types/data.rs` |
| TrainingConfig | `src/config/types/training.rs` |
| dataset load/split | `src/data/dataset/core.rs` |
| manifest / label schema | `src/data/parser/manifest.rs` |
| affinity label parser | `src/data/parser/affinity.rs` |
| PDBbind-like discovery | `src/data/parser/discovery.rs` |
| SDF parser | `src/data/parser/sdf.rs` |
| feature builders | `src/data/features/builders.rs` |
| trainer | `src/training/trainer.rs` |
| stage scheduler | `src/training/scheduler.rs` |
| primary objectives | `src/losses/task.rs` |
| training CLI entrypoint | `src/training/entrypoints.rs` |
| unseen-pocket experiment | `src/experiments/unseen_pocket/run.rs` |

## 10. 外部来源备注

- PDBbind++ 2020 Hugging Face dataset card 说明 refined set 约 5,316 个 protein-ligand complexes，目录按 PDB ID 组织，并包含 protein/pocket/ligand 结构文件。
- Zenodo 上存在 reprocessed PDBbind-v2020 数据包，可作为 PDBbind-like 结构来源之一；使用前应确认其预处理协议、许可和引用要求是否符合目标实验。
- PDBbind 官方数据应遵守官方注册、下载和引用要求；本仓库只提供 parser 和整理脚本，不提供绕过授权的数据下载器。
