use tch::{Kind, Tensor};

pub(crate) struct LigandPocketSlotAttentionBias {
    pub values: Tensor,
    pub chemistry_role_coverage: f64,
}

pub(crate) fn ligand_pocket_slot_attention_bias_with_scale(
    ligand_coords: &Tensor,
    ligand_mask: &Tensor,
    pocket_coords: &Tensor,
    pocket_mask: &Tensor,
    ligand_chemistry_roles: Option<&Tensor>,
    pocket_chemistry_roles: Option<&Tensor>,
    ligand_slot_weights: Option<&Tensor>,
    pocket_slot_weights: Option<&Tensor>,
    ligand_slots: i64,
    pocket_slots: i64,
    chemistry_role_scale: f64,
) -> LigandPocketSlotAttentionBias {
    let batch = ligand_coords.size()[0];
    if ligand_slots <= 0 || pocket_slots <= 0 {
        return LigandPocketSlotAttentionBias {
            values: Tensor::zeros(
                [batch, ligand_slots, pocket_slots],
                (Kind::Float, ligand_coords.device()),
            ),
            chemistry_role_coverage: 0.0,
        };
    }

    let pair_mask = ligand_mask.unsqueeze(2) * pocket_mask.unsqueeze(1);
    let diffs = ligand_coords.unsqueeze(2) - pocket_coords.unsqueeze(1);
    let distances = diffs
        .pow_tensor_scalar(2.0)
        .sum_dim_intlist([3].as_slice(), false, Kind::Float)
        .sqrt();
    let valid_distances = distances.shallow_clone() * &pair_mask;
    let mean_distance = valid_distances.sum_dim_intlist([1, 2].as_slice(), false, Kind::Float)
        / pair_mask
            .sum_dim_intlist([1, 2].as_slice(), false, Kind::Float)
            .clamp_min(1.0);
    let contact_fraction = distances.lt(4.5).to_kind(Kind::Float) * &pair_mask;
    let contact_fraction = contact_fraction.sum_dim_intlist([1, 2].as_slice(), false, Kind::Float)
        / pair_mask
            .sum_dim_intlist([1, 2].as_slice(), false, Kind::Float)
            .clamp_min(1.0);
    let scalar_bias = contact_fraction - mean_distance.clamp_max(12.0) / 12.0;
    let base_bias = scalar_bias
        .reshape([batch, 1, 1])
        .expand([batch, ligand_slots, pocket_slots], true);

    let Some(ligand_roles) = ligand_chemistry_roles else {
        return LigandPocketSlotAttentionBias {
            values: base_bias,
            chemistry_role_coverage: 0.0,
        };
    };
    let Some(pocket_roles) = pocket_chemistry_roles else {
        return LigandPocketSlotAttentionBias {
            values: base_bias,
            chemistry_role_coverage: 0.0,
        };
    };

    let (role_component, chemistry_role_coverage) = ligand_pocket_role_bias_component(
        ligand_roles,
        pocket_roles,
        &pair_mask,
        &distances,
        ligand_slot_weights,
        pocket_slot_weights,
        ligand_slots,
        pocket_slots,
    );
    let values = if chemistry_role_scale > 0.0 {
        (base_bias + role_component * chemistry_role_scale).clamp(-2.0, 2.0)
    } else {
        base_bias
    };

    LigandPocketSlotAttentionBias {
        values,
        chemistry_role_coverage,
    }
}

fn ligand_pocket_role_bias_component(
    ligand_roles: &Tensor,
    pocket_roles: &Tensor,
    pair_mask: &Tensor,
    distances: &Tensor,
    ligand_slot_weights: Option<&Tensor>,
    pocket_slot_weights: Option<&Tensor>,
    ligand_slots: i64,
    pocket_slots: i64,
) -> (Tensor, f64) {
    let batch = pair_mask.size()[0];
    if role_tensor_is_unusable(ligand_roles) || role_tensor_is_unusable(pocket_roles) {
        return (
            Tensor::zeros(
                [batch, ligand_slots, pocket_slots],
                (Kind::Float, pair_mask.device()),
            ),
            0.0,
        );
    }

    let ligand_available = role_channel(ligand_roles, ROLE_AVAILABLE);
    let pocket_available = role_channel(pocket_roles, ROLE_AVAILABLE);
    let role_pair_mask = pair_product(&ligand_available, &pocket_available) * pair_mask;
    let valid_pair_count = pair_mask
        .sum_dim_intlist([1, 2].as_slice(), false, Kind::Float)
        .clamp_min(1.0);
    let role_pair_count = role_pair_mask.sum_dim_intlist([1, 2].as_slice(), false, Kind::Float);
    let chemistry_role_coverage = (&role_pair_count / &valid_pair_count)
        .mean(Kind::Float)
        .double_value(&[]);

    let hbond = pair_product(
        &role_channel(ligand_roles, ROLE_DONOR),
        &role_channel(pocket_roles, ROLE_ACCEPTOR),
    ) + pair_product(
        &role_channel(ligand_roles, ROLE_ACCEPTOR),
        &role_channel(pocket_roles, ROLE_DONOR),
    );
    let hydrophobic = pair_product(
        &role_channel(ligand_roles, ROLE_HYDROPHOBIC),
        &role_channel(pocket_roles, ROLE_HYDROPHOBIC),
    );
    let aromatic = pair_product(
        &role_channel(ligand_roles, ROLE_AROMATIC),
        &role_channel(pocket_roles, ROLE_AROMATIC),
    );
    let charge = pair_product(
        &role_channel(ligand_roles, ROLE_POSITIVE),
        &role_channel(pocket_roles, ROLE_NEGATIVE),
    ) + pair_product(
        &role_channel(ligand_roles, ROLE_NEGATIVE),
        &role_channel(pocket_roles, ROLE_POSITIVE),
    );
    let metal = pair_product(
        &role_channel(ligand_roles, ROLE_METAL_BINDING),
        &role_channel(pocket_roles, ROLE_METAL_BINDING),
    );
    let compatible_pairs = (hbond + hydrophobic + aromatic + charge + metal).clamp(0.0, 1.5);
    let mean_compatibility =
        (compatible_pairs * &role_pair_mask).sum_dim_intlist([1, 2].as_slice(), false, Kind::Float)
            / role_pair_count.clamp_min(1.0);

    let clash_fraction = (distances.lt(1.5).to_kind(Kind::Float) * pair_mask).sum_dim_intlist(
        [1, 2].as_slice(),
        false,
        Kind::Float,
    ) / valid_pair_count;
    let role_score = (mean_compatibility - clash_fraction * 0.75).clamp(-1.0, 1.0);
    let slot_profile = slot_pair_profile(
        ligand_slot_weights,
        pocket_slot_weights,
        batch,
        ligand_slots,
        pocket_slots,
        pair_mask.device(),
    );
    let role_component = role_score.reshape([batch, 1, 1]) * (slot_profile * 0.5 + 0.5);
    (role_component, chemistry_role_coverage)
}

pub(super) fn role_tensor_is_unusable(roles: &Tensor) -> bool {
    let size = roles.size();
    size.len() < 3 || size[2] <= ROLE_AVAILABLE
}

pub(super) fn role_channel(roles: &Tensor, channel: i64) -> Tensor {
    roles.narrow(2, channel, 1).to_kind(Kind::Float)
}

fn pair_product(ligand_channel: &Tensor, pocket_channel: &Tensor) -> Tensor {
    ligand_channel * pocket_channel.transpose(1, 2)
}

pub(super) fn slot_pair_profile(
    ligand_slot_weights: Option<&Tensor>,
    pocket_slot_weights: Option<&Tensor>,
    batch: i64,
    ligand_slots: i64,
    pocket_slots: i64,
    device: tch::Device,
) -> Tensor {
    let Some(ligand_weights) = ligand_slot_weights else {
        return Tensor::ones([batch, ligand_slots, pocket_slots], (Kind::Float, device));
    };
    let Some(pocket_weights) = pocket_slot_weights else {
        return Tensor::ones([batch, ligand_slots, pocket_slots], (Kind::Float, device));
    };
    let ligand_size = ligand_weights.size();
    let pocket_size = pocket_weights.size();
    if ligand_size.len() < 2
        || pocket_size.len() < 2
        || ligand_size[0] != batch
        || pocket_size[0] != batch
        || ligand_size[1] != ligand_slots
        || pocket_size[1] != pocket_slots
    {
        return Tensor::ones([batch, ligand_slots, pocket_slots], (Kind::Float, device));
    }

    let outer = ligand_weights.to_kind(Kind::Float).unsqueeze(2)
        * pocket_weights.to_kind(Kind::Float).unsqueeze(1);
    let max_per_example = outer.amax([1, 2].as_slice(), true).clamp_min(1e-6);
    outer / max_per_example
}

pub(super) const ROLE_DONOR: i64 = 0;
pub(super) const ROLE_ACCEPTOR: i64 = 1;
pub(super) const ROLE_HYDROPHOBIC: i64 = 2;
pub(super) const ROLE_AROMATIC: i64 = 3;
pub(super) const ROLE_POSITIVE: i64 = 4;
pub(super) const ROLE_NEGATIVE: i64 = 5;
pub(super) const ROLE_METAL_BINDING: i64 = 6;
pub(super) const ROLE_AVAILABLE: i64 = 8;
