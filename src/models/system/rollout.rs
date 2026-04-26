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

