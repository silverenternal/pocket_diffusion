//! Legacy demo runner and terminal presentation helpers.

use tch::{nn, Device};

use crate::{
    legacy::PocketDiffusionPipeline,
    pocket::{create_example_prrsv_pocket, PocketFeatureExtractor},
    types::{AtomType, CandidateMolecule},
};

/// Run the legacy pocket-conditioned generation demo and print a CLI report.
pub fn run_legacy_demo(num_candidates: usize, top_k: usize, modular_bridge: bool) {
    println!("================================================");
    println!("  基于口袋条件扩散的结构感知小分子生成系统");
    println!("  Pocket-Conditioned Diffusion Molecule Gen");
    println!("================================================");

    println!("\n配置:");
    println!("  - 候选分子数量: {}", num_candidates);
    println!("  - Top-K筛选: {}", top_k);
    println!(
        "  - 生成后端: {}",
        if modular_bridge {
            "modular research bridge"
        } else {
            "legacy generator"
        }
    );

    println!("\n[1/4] 初始化神经网络模型...");
    let vs = nn::VarStore::new(Device::Cpu);
    let pipeline = PocketDiffusionPipeline::new(&vs.root());

    println!("\n[2/4] 加载目标蛋白口袋 (PRRSV核衣壳蛋白)...");
    let pocket = create_example_prrsv_pocket();
    let embedding = PocketFeatureExtractor::extract(&pocket);

    println!("  口袋原子数: {}", embedding.total_atoms);
    println!("  重原子数: {}", embedding.heavy_atoms);
    println!("  碳原子数: {}", embedding.carbon_count);
    println!("  氮原子数: {}", embedding.nitrogen_count);
    println!("  氧原子数: {}", embedding.oxygen_count);
    println!("  硫原子数: {}", embedding.sulfur_count);
    println!("  口袋半径: {:.2} Å", embedding.pocket_radius);
    println!("  坐标标准差: {:.2} Å", embedding.coord_std);

    println!("\n[3/4] 生成候选分子集...");
    let result = if modular_bridge {
        pipeline.generate_and_rank_with_modular_bridge(&pocket, num_candidates, top_k)
    } else {
        pipeline.generate_and_rank(&pocket, num_candidates, top_k)
    };

    println!("  生成候选分子: {} 个", result.candidates.len());
    println!(
        "  平均原子数: {:.1}",
        result
            .candidates
            .iter()
            .map(|candidate| candidate.ligand.atoms.len() as f64)
            .sum::<f64>()
            / result.candidates.len() as f64
    );

    println!("\n[4/4] Top-{} 亲和力最高的分子:", top_k);
    println!("--------------------------------");

    for (index, candidate) in result.top_candidates.iter().enumerate() {
        print_candidate_summary(index, candidate);
    }
}

fn print_candidate_summary(index: usize, candidate: &CandidateMolecule) {
    let affinity = candidate.affinity_score.unwrap_or(0.0);
    let num_atoms = candidate.ligand.atoms.len();
    let num_bonds = candidate.ligand.bonds.len();

    println!("\n排名 #{}:", index + 1);
    println!("  结合亲和力: {:.2} kcal/mol", affinity);
    println!("  原子数量: {}", num_atoms);
    println!("  化学键数量: {}", num_bonds);

    let mut c_count = 0;
    let mut n_count = 0;
    let mut o_count = 0;
    let mut s_count = 0;
    let mut h_count = 0;

    for atom in &candidate.ligand.atoms {
        match atom.atom_type {
            AtomType::Carbon => c_count += 1,
            AtomType::Nitrogen => n_count += 1,
            AtomType::Oxygen => o_count += 1,
            AtomType::Sulfur => s_count += 1,
            AtomType::Hydrogen => h_count += 1,
            _ => {}
        }
    }

    println!(
        "  原子组成: C({}) N({}) O({}) S({}) H({})",
        c_count, n_count, o_count, s_count, h_count
    );

    println!("  前5个原子坐标:");
    for (atom_index, atom) in candidate.ligand.atoms.iter().take(5).enumerate() {
        println!(
            "    [{}] ({:?}) [{:7.3}, {:7.3}, {:7.3}]",
            atom_index, atom.atom_type, atom.coords[0], atom.coords[1], atom.coords[2]
        );
    }
}
