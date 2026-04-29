impl Phase1ResearchSystem {
    fn skipped_rollout_diagnostics_record(
        &self,
        initial_state: &ConditionedGenerationState,
        raw_path_means: GenerationGateSummary,
    ) -> GenerationRolloutRecord {
        GenerationRolloutRecord {
            example_id: initial_state.example_id.clone(),
            protein_id: initial_state.protein_id.clone(),
            generation_mode: self.generation_mode.as_str().to_string(),
            decoder_capability: self.decoder_capability_label().to_string(),
            atom_count_source: self.generation_mode.atom_count_source_label().to_string(),
            atom_count_prior_provenance: self.atom_count_prior_provenance_label().to_string(),
            topology_source: self.generation_mode.topology_source_label().to_string(),
            geometry_source: self.generation_mode.geometry_source_label().to_string(),
            conditioning_coordinate_frame: self.conditioning_coordinate_frame_label().to_string(),
            flow_x0_source: None,
            configured_steps: self.generation_target.rollout_steps,
            executed_steps: 0,
            stopped_early: false,
            path_usage: path_usage_summary(raw_path_means, &[]),
            context_refresh_policy: self.generation_target.context_refresh_policy.label(),
            refresh_count: 0,
            last_refresh_step: None,
            stale_context_steps: 0,
            severe_clash_flag: false,
            valence_guardrail_flag: false,
            pharmacophore_conflict_flag: false,
            guardrail_blockable_stop_flag: false,
            steps: Vec::new(),
        }
    }

    fn rollout_generation(
        &self,
        example: &MolecularExample,
        initial_state: &ConditionedGenerationState,
        raw_path_means: GenerationGateSummary,
        interaction_execution_context: &InteractionExecutionContext,
    ) -> GenerationRolloutRecord {
        if self.generation_backend_family == GenerationBackendFamilyConfig::FlowMatching {
            return self.rollout_flow_matching(
                example,
                initial_state,
                raw_path_means,
                interaction_execution_context,
            );
        }
        let mut state = initial_state.clone();
        let mut steps = Vec::with_capacity(self.generation_target.rollout_steps);
        let mut stopped_early = false;
        let mut stable_steps = 0_usize;
        let mut previous_coord_delta: Option<Tensor> = None;
        let mut previous_atom_logits: Option<Tensor> = None;
        let mut rollout_guardrails = RolloutGuardrailFlags::default();
        let mut rollout_guardrail_blockable_stop = false;
        let mut current_gate_summary = raw_path_means;
        let mut step_gate_summaries = Vec::with_capacity(self.generation_target.rollout_steps);

        for step_index in 0..self.generation_target.rollout_steps {
            state.partial_ligand.step_index = step_index as i64;
            if self
                .generation_target
                .context_refresh_policy
                .should_refresh_at_step(step_index)
            {
                current_gate_summary = self.refresh_generation_context(
                    example,
                    &mut state,
                    step_index,
                    interaction_execution_context,
                );
            }
            let decoded = self.generator_stack.ligand_decoder.decode(&state);
            let stop_probability = decoded
                .stop_logit
                .sigmoid()
                .mean(Kind::Float)
                .double_value(&[]);
            let (next_atom_types, atom_change_fraction, updated_atom_logits) = self
                .next_atom_state(
                    &state.partial_ligand.atom_types,
                    &decoded.atom_type_logits,
                    previous_atom_logits.as_ref(),
                    step_index,
                );
            let (next_coords, mean_displacement, coordinate_step_scale, updated_coord_delta) = self
                .next_coordinate_state(
                    example,
                    &state.partial_ligand.coords,
                    &decoded.coordinate_deltas,
                    step_index,
                    previous_coord_delta.as_ref(),
                );
            previous_atom_logits = updated_atom_logits;
            previous_coord_delta = updated_coord_delta;
            let stable_now = mean_displacement <= self.generation_target.stop_delta_threshold
                && atom_change_fraction <= self.generation_target.stop_delta_threshold;
            stable_steps = if stable_now { stable_steps + 1 } else { 0 };
            let guardrails =
                self.rollout_guardrail_flags_for_generation(example, &next_atom_types, &next_coords);
            let stop_ready = step_index + 1 >= self.generation_target.min_rollout_steps;
            let should_stop = stop_ready
                && match self.generation_target.rollout_mode {
                    GenerationRolloutMode::Lightweight => {
                        stop_probability >= self.generation_target.stop_probability_threshold
                    }
                    GenerationRolloutMode::MomentumRefine => {
                        stop_probability >= self.generation_target.stop_probability_threshold
                            || stable_steps >= self.generation_target.stop_patience
                    }
                };
            let guardrail_blockable_stop_flag = should_stop && guardrails.any();
            rollout_guardrails.merge(guardrails);
            rollout_guardrail_blockable_stop |= guardrail_blockable_stop_flag;
            step_gate_summaries.push(current_gate_summary);

            steps.push(GenerationStepRecord {
                step_index,
                stop_probability,
                stopped: should_stop,
                atom_types: tensor_to_i64_vec(&next_atom_types),
                coords: tensor_to_coords(&next_coords),
                native_bonds: Vec::new(),
                native_bond_types: Vec::new(),
                constrained_native_bonds: Vec::new(),
                constrained_native_bond_types: Vec::new(),
                native_graph_provenance: NativeGraphLayerProvenance::default(),
                mean_displacement,
                atom_change_fraction,
                coordinate_step_scale,
                severe_clash_flag: guardrails.severe_clash,
                valence_guardrail_flag: guardrails.valence_guardrail,
                pharmacophore_conflict_flag: guardrails.pharmacophore_conflict,
                guardrail_blockable_stop_flag,
                flow_diagnostics: std::collections::BTreeMap::new(),
            });

            state.partial_ligand.atom_types = next_atom_types;
            state.partial_ligand.coords = next_coords;

            if should_stop {
                stopped_early = true;
                break;
            }
        }

        if steps.is_empty() {
            let guardrails = rollout_guardrail_flags(
                example,
                &example.decoder_supervision.corrupted_atom_types,
                &example.decoder_supervision.noisy_coords,
            );
            rollout_guardrails.merge(guardrails);
            steps.push(GenerationStepRecord {
                step_index: 0,
                stop_probability: 0.0,
                stopped: false,
                atom_types: tensor_to_i64_vec(&example.decoder_supervision.corrupted_atom_types),
                coords: tensor_to_coords(&example.decoder_supervision.noisy_coords),
                native_bonds: Vec::new(),
                native_bond_types: Vec::new(),
                constrained_native_bonds: Vec::new(),
                constrained_native_bond_types: Vec::new(),
                native_graph_provenance: NativeGraphLayerProvenance::default(),
                mean_displacement: 0.0,
                atom_change_fraction: 0.0,
                coordinate_step_scale: self.generation_target.coordinate_step_scale,
                severe_clash_flag: guardrails.severe_clash,
                valence_guardrail_flag: guardrails.valence_guardrail,
                pharmacophore_conflict_flag: guardrails.pharmacophore_conflict,
                guardrail_blockable_stop_flag: false,
                flow_diagnostics: std::collections::BTreeMap::new(),
            });
            step_gate_summaries.push(raw_path_means);
        }

        GenerationRolloutRecord {
            example_id: initial_state.example_id.clone(),
            protein_id: initial_state.protein_id.clone(),
            generation_mode: self.generation_mode.as_str().to_string(),
            decoder_capability: self.decoder_capability_label().to_string(),
            atom_count_source: self.generation_mode.atom_count_source_label().to_string(),
            atom_count_prior_provenance: self.atom_count_prior_provenance_label().to_string(),
            topology_source: self.generation_mode.topology_source_label().to_string(),
            geometry_source: self.generation_mode.geometry_source_label().to_string(),
            conditioning_coordinate_frame: self.conditioning_coordinate_frame_label().to_string(),
            flow_x0_source: None,
            configured_steps: self.generation_target.rollout_steps,
            executed_steps: steps.len(),
            stopped_early,
            path_usage: path_usage_summary(raw_path_means, &step_gate_summaries),
            context_refresh_policy: self.generation_target.context_refresh_policy.label(),
            refresh_count: refresh_count_for_policy(
                &self.generation_target.context_refresh_policy,
                steps.len(),
            ),
            last_refresh_step: last_refresh_step_for_policy(
                &self.generation_target.context_refresh_policy,
                steps.len(),
            ),
            stale_context_steps: stale_context_steps_for_policy(
                &self.generation_target.context_refresh_policy,
                steps.len(),
            ),
            severe_clash_flag: rollout_guardrails.severe_clash,
            valence_guardrail_flag: rollout_guardrails.valence_guardrail,
            pharmacophore_conflict_flag: rollout_guardrails.pharmacophore_conflict,
            guardrail_blockable_stop_flag: rollout_guardrail_blockable_stop,
            steps,
        }
    }

    fn next_atom_state(
        &self,
        current_atom_types: &Tensor,
        atom_type_logits: &Tensor,
        previous_atom_logits: Option<&Tensor>,
        step_index: usize,
    ) -> (Tensor, f64, Option<Tensor>) {
        if atom_type_logits.numel() == 0 {
            return (
                current_atom_types.shallow_clone(),
                0.0,
                previous_atom_logits.map(Tensor::shallow_clone),
            );
        }

        let committed_logits = match self.generation_target.rollout_mode {
            GenerationRolloutMode::Lightweight => atom_type_logits.shallow_clone(),
            GenerationRolloutMode::MomentumRefine => {
                let blended = if let Some(previous) = previous_atom_logits {
                    previous * self.generation_target.atom_momentum
                        + atom_type_logits * (1.0 - self.generation_target.atom_momentum)
                } else {
                    atom_type_logits.shallow_clone()
                };
                blended / self.generation_target.atom_commit_temperature
            }
        };
        let next_atom_types = if self.generation_target.sampling_temperature > 0.0 {
            sample_atom_types(
                &committed_logits,
                self.generation_target.sampling_temperature,
                self.generation_target.sampling_top_k,
                self.generation_target.sampling_top_p,
                self.generation_target.sampling_seed,
                step_index,
            )
        } else {
            committed_logits.argmax(-1, false)
        };
        let atom_change_fraction = atom_change_fraction(current_atom_types, &next_atom_types);
        (
            next_atom_types,
            atom_change_fraction,
            Some(committed_logits),
        )
    }

    fn next_coordinate_state(
        &self,
        example: &MolecularExample,
        current_coords: &Tensor,
        coordinate_deltas: &Tensor,
        step_index: usize,
        previous_coord_delta: Option<&Tensor>,
    ) -> (Tensor, f64, f64, Option<Tensor>) {
        if coordinate_deltas.numel() == 0 {
            return (
                current_coords.shallow_clone(),
                0.0,
                self.generation_target.coordinate_step_scale,
                previous_coord_delta.map(Tensor::shallow_clone),
            );
        }

        let normalized_delta = clip_coordinate_delta_norm(
            coordinate_deltas,
            self.generation_target.max_coordinate_delta_norm,
        );
        let effective_delta = match self.generation_target.rollout_mode {
            GenerationRolloutMode::Lightweight => normalized_delta,
            GenerationRolloutMode::MomentumRefine => {
                if let Some(previous) = previous_coord_delta {
                    previous * self.generation_target.coordinate_momentum
                        + normalized_delta * (1.0 - self.generation_target.coordinate_momentum)
                } else {
                    normalized_delta
                }
            }
        };
        let anneal = match self.generation_target.rollout_mode {
            GenerationRolloutMode::Lightweight => 1.0,
            GenerationRolloutMode::MomentumRefine => {
                let fraction =
                    step_index as f64 / self.generation_target.rollout_steps.max(1) as f64;
                (1.0 - 0.35 * fraction).max(0.5)
            }
        };
        let coordinate_step_scale = self.generation_target.coordinate_step_scale * anneal;
        let scaled_delta = &effective_delta * coordinate_step_scale;
        let pocket_guidance = pocket_guidance_delta(
            current_coords,
            &example.pocket.coords,
            coordinate_step_scale,
            step_index,
            self.generation_target.rollout_steps,
        ) * self.generation_target.pocket_guidance_scale;
        let sampling_noise = deterministic_coordinate_noise(
            current_coords,
            self.generation_target.coordinate_sampling_noise_std,
            self.generation_target.sampling_seed,
            step_index,
        );
        let effective_update = &scaled_delta + &pocket_guidance + &sampling_noise;
        let unconstrained_next = current_coords + &effective_update;
        let next_coords = constrain_to_pocket_envelope(
            &unconstrained_next,
            &example.pocket.coords,
            self.generation_target.pocket_guidance_scale,
        );
        let realized_update = &next_coords - current_coords;
        let mean_displacement = per_atom_displacement_mean(&realized_update);
        (
            next_coords,
            mean_displacement,
            coordinate_step_scale,
            Some((&realized_update / coordinate_step_scale.max(1e-6)).detach()),
        )
    }
}

pub(super) fn gate_summary_from_interaction_diagnostics(
    diagnostics: &crate::models::interaction::CrossModalInteractionDiagnostics,
) -> GenerationGateSummary {
    GenerationGateSummary {
        topo_from_geo: diagnostics.topo_from_geo.gate_mean,
        topo_from_pocket: diagnostics.topo_from_pocket.gate_mean,
        geo_from_topo: diagnostics.geo_from_topo.gate_mean,
        geo_from_pocket: diagnostics.geo_from_pocket.gate_mean,
        pocket_from_topo: diagnostics.pocket_from_topo.gate_mean,
        pocket_from_geo: diagnostics.pocket_from_geo.gate_mean,
    }
}

pub(super) fn path_usage_summary(
    raw_path_means: GenerationGateSummary,
    step_gate_summaries: &[GenerationGateSummary],
) -> GenerationPathUsageSummary {
    let step_bucketed_path_means = step_gate_summaries
        .iter()
        .copied()
        .enumerate()
        .map(|(step_index, path_means)| GenerationStepPathUsageSummary {
            start_step: step_index,
            end_step: step_index,
            path_means,
        })
        .collect();

    GenerationPathUsageSummary {
        raw_path_means,
        step_bucketed_path_means,
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct RolloutGuardrailFlags {
    pub(super) severe_clash: bool,
    pub(super) valence_guardrail: bool,
    pub(super) pharmacophore_conflict: bool,
}

impl RolloutGuardrailFlags {
    fn any(self) -> bool {
        self.severe_clash || self.valence_guardrail || self.pharmacophore_conflict
    }

    pub(super) fn merge(&mut self, other: Self) {
        self.severe_clash |= other.severe_clash;
        self.valence_guardrail |= other.valence_guardrail;
        self.pharmacophore_conflict |= other.pharmacophore_conflict;
    }
}

impl Phase1ResearchSystem {
    pub(super) fn rollout_guardrail_flags_for_generation(
        &self,
        example: &MolecularExample,
        atom_types: &Tensor,
        coords: &Tensor,
    ) -> RolloutGuardrailFlags {
        if self.generation_mode.uses_target_ligand_initialization() {
            return rollout_guardrail_flags(example, atom_types, coords);
        }
        let generated_adjacency =
            scaffold_adjacency_from_coords(coords, self.flow_matching_config.noise_scale);
        rollout_guardrail_flags_with_adjacency(example, atom_types, coords, &generated_adjacency)
    }
}

pub(super) fn rollout_guardrail_flags(
    example: &MolecularExample,
    atom_types: &Tensor,
    coords: &Tensor,
) -> RolloutGuardrailFlags {
    rollout_guardrail_flags_with_adjacency(example, atom_types, coords, &example.topology.adjacency)
}

fn rollout_guardrail_flags_with_adjacency(
    example: &MolecularExample,
    atom_types: &Tensor,
    coords: &Tensor,
    adjacency: &Tensor,
) -> RolloutGuardrailFlags {
    RolloutGuardrailFlags {
        severe_clash: has_severe_pocket_clash(coords, &example.pocket.coords),
        valence_guardrail: has_valence_guardrail_violation(atom_types, adjacency),
        pharmacophore_conflict: has_pharmacophore_conflict(
            atom_types,
            coords,
            &example.pocket.coords,
            &example.pocket.chemistry_roles.role_vectors,
        ),
    }
}

fn has_severe_pocket_clash(ligand_coords: &Tensor, pocket_coords: &Tensor) -> bool {
    if ligand_coords.numel() == 0 || pocket_coords.numel() == 0 {
        return false;
    }
    let distances = ligand_pocket_distances(ligand_coords, pocket_coords);
    distances.min().double_value(&[]) < 1.25
}

fn has_valence_guardrail_violation(atom_types: &Tensor, adjacency: &Tensor) -> bool {
    let atom_count = atom_types
        .size()
        .first()
        .copied()
        .unwrap_or(0)
        .min(adjacency.size().first().copied().unwrap_or(0))
        .max(0);
    for atom_index in 0..atom_count {
        let degree = (0..atom_count)
            .filter(|neighbor| adjacency.double_value(&[atom_index, *neighbor]) > 0.5)
            .count();
        if degree > max_reasonable_valence(atom_types.int64_value(&[atom_index])) {
            return true;
        }
    }
    false
}

fn has_pharmacophore_conflict(
    atom_types: &Tensor,
    ligand_coords: &Tensor,
    pocket_coords: &Tensor,
    pocket_roles: &Tensor,
) -> bool {
    if atom_types.numel() == 0
        || ligand_coords.numel() == 0
        || pocket_coords.numel() == 0
        || pocket_roles.numel() == 0
    {
        return false;
    }
    let ligand_roles = chemistry_roles_from_atom_type_tensor(atom_types);
    let ligand_roles = ligand_roles.role_vectors;
    let ligand_count = ligand_coords.size().first().copied().unwrap_or(0).max(0);
    let pocket_count = pocket_coords
        .size()
        .first()
        .copied()
        .unwrap_or(0)
        .min(pocket_roles.size().first().copied().unwrap_or(0))
        .max(0);

    for ligand_index in 0..ligand_count {
        if role_value(&ligand_roles, ligand_index, ROLE_AVAILABLE) < 0.5 {
            continue;
        }
        for pocket_index in 0..pocket_count {
            if role_value(pocket_roles, pocket_index, ROLE_AVAILABLE) < 0.5 {
                continue;
            }
            if coordinate_distance(ligand_coords, ligand_index, pocket_coords, pocket_index) > 4.0 {
                continue;
            }
            let same_charge = role_value(&ligand_roles, ligand_index, ROLE_POSITIVE)
                * role_value(pocket_roles, pocket_index, ROLE_POSITIVE)
                + role_value(&ligand_roles, ligand_index, ROLE_NEGATIVE)
                    * role_value(pocket_roles, pocket_index, ROLE_NEGATIVE);
            if same_charge > 0.2 {
                return true;
            }
        }
    }
    false
}

fn ligand_pocket_distances(ligand_coords: &Tensor, pocket_coords: &Tensor) -> Tensor {
    let diffs = ligand_coords.unsqueeze(1) - pocket_coords.unsqueeze(0);
    diffs
        .pow_tensor_scalar(2.0)
        .sum_dim_intlist([2].as_slice(), false, Kind::Float)
        .clamp_min(1e-12)
        .sqrt()
}

fn coordinate_distance(left: &Tensor, left_index: i64, right: &Tensor, right_index: i64) -> f64 {
    let dx = left.double_value(&[left_index, 0]) - right.double_value(&[right_index, 0]);
    let dy = left.double_value(&[left_index, 1]) - right.double_value(&[right_index, 1]);
    let dz = left.double_value(&[left_index, 2]) - right.double_value(&[right_index, 2]);
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn role_value(roles: &Tensor, row: i64, channel: i64) -> f64 {
    if roles.size().len() < 2
        || row >= roles.size()[0]
        || channel >= roles.size()[1]
        || row < 0
        || channel < 0
    {
        return 0.0;
    }
    roles.double_value(&[row, channel])
}

fn max_reasonable_valence(atom_type: i64) -> usize {
    match atom_type {
        0 => 4,
        1 => 3,
        2 => 2,
        3 => 6,
        4 => 1,
        _ => 4,
    }
}

const ROLE_POSITIVE: i64 = 4;
const ROLE_NEGATIVE: i64 = 5;
const ROLE_AVAILABLE: i64 = 8;

pub(super) fn refresh_count_for_policy(
    policy: &InferenceContextRefreshPolicy,
    executed_steps: usize,
) -> usize {
    (0..executed_steps)
        .filter(|step| policy.should_refresh_at_step(*step))
        .count()
}

pub(super) fn last_refresh_step_for_policy(
    policy: &InferenceContextRefreshPolicy,
    executed_steps: usize,
) -> Option<usize> {
    (0..executed_steps)
        .rev()
        .find(|step| policy.should_refresh_at_step(*step))
}

pub(super) fn stale_context_steps_for_policy(
    policy: &InferenceContextRefreshPolicy,
    executed_steps: usize,
) -> usize {
    executed_steps.saturating_sub(refresh_count_for_policy(policy, executed_steps))
}

fn sample_atom_types(
    logits: &Tensor,
    temperature: f64,
    top_k: usize,
    top_p: f64,
    seed: u64,
    step_index: usize,
) -> Tensor {
    let shape = logits.size();
    let atom_count = shape.first().copied().unwrap_or(0).max(0) as usize;
    let vocab = shape.get(1).copied().unwrap_or(0).max(0) as usize;
    if atom_count == 0 || vocab == 0 {
        return logits.argmax(-1, false);
    }

    let mut sampled = Vec::with_capacity(atom_count);
    for atom_ix in 0..atom_count {
        let mut probabilities = (0..vocab)
            .map(|token_ix| {
                let value =
                    logits.double_value(&[atom_ix as i64, token_ix as i64]) / temperature.max(1e-6);
                (token_ix as i64, value)
            })
            .collect::<Vec<_>>();
        probabilities.sort_by(|left, right| {
            right
                .1
                .partial_cmp(&left.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let keep = if top_k == 0 {
            probabilities.len()
        } else {
            top_k.min(probabilities.len())
        };
        probabilities.truncate(keep.max(1));

        let max_logit = probabilities
            .iter()
            .map(|(_, value)| *value)
            .fold(f64::NEG_INFINITY, f64::max);
        let mut weighted = probabilities
            .into_iter()
            .map(|(token_ix, value)| (token_ix, (value - max_logit).exp()))
            .collect::<Vec<_>>();
        let total = weighted
            .iter()
            .map(|(_, weight)| *weight)
            .sum::<f64>()
            .max(1e-12);
        for (_, weight) in &mut weighted {
            *weight /= total;
        }
        weighted.sort_by(|left, right| {
            right
                .1
                .partial_cmp(&left.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut cumulative = 0.0;
        let mut filtered = Vec::new();
        for (token_ix, probability) in weighted {
            cumulative += probability;
            filtered.push((token_ix, probability));
            if cumulative >= top_p {
                break;
            }
        }
        let filtered_total = filtered
            .iter()
            .map(|(_, probability)| *probability)
            .sum::<f64>()
            .max(1e-12);
        let draw = deterministic_unit(seed, step_index, atom_ix, 0);
        let mut running = 0.0;
        let mut picked = filtered.last().map(|(token_ix, _)| *token_ix).unwrap_or(0);
        for (token_ix, probability) in filtered {
            running += probability / filtered_total;
            if draw <= running {
                picked = token_ix;
                break;
            }
        }
        sampled.push(picked);
    }
    Tensor::from_slice(&sampled)
        .to_kind(Kind::Int64)
        .to_device(logits.device())
}

fn deterministic_coordinate_noise(
    coords: &Tensor,
    std: f64,
    seed: u64,
    step_index: usize,
) -> Tensor {
    if std <= 0.0 || coords.numel() == 0 {
        return Tensor::zeros_like(coords);
    }
    let shape = coords.size();
    let atom_count = shape.first().copied().unwrap_or(0).max(0) as usize;
    let mut values = Vec::with_capacity(atom_count * 3);
    for atom_ix in 0..atom_count {
        for axis in 0..3 {
            let centered = deterministic_unit(seed, step_index, atom_ix, axis + 1) * 2.0 - 1.0;
            values.push((centered * std) as f32);
        }
    }
    Tensor::from_slice(&values)
        .reshape([atom_count as i64, 3])
        .to_device(coords.device())
}

fn deterministic_unit(seed: u64, step_index: usize, atom_ix: usize, stream: usize) -> f64 {
    let mut value = seed
        ^ ((step_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
        ^ ((atom_ix as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9))
        ^ ((stream as u64).wrapping_mul(0x94D0_49BB_1331_11EB));
    value ^= value >> 30;
    value = value.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^= value >> 31;
    (value as f64) / (u64::MAX as f64)
}
