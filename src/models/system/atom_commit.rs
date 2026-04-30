use tch::{Kind, Tensor};

const BASE_ATOM_TYPE_COMMIT_CONFIDENCE_THRESHOLD: f64 = 0.85;
const HEAVY_TO_HYDROGEN_COMMIT_CONFIDENCE_THRESHOLD: f64 = 0.97;
const HYDROGEN_TO_HEAVY_COMMIT_CONFIDENCE_THRESHOLD: f64 = 0.70;
const HYDROGEN_TOKEN: i64 = 4;

pub(super) fn confidence_gated_atom_type_commit(
    logits: &Tensor,
    previous_atom_types: &Tensor,
) -> Tensor {
    let atom_count = previous_atom_types
        .size()
        .first()
        .copied()
        .unwrap_or(0)
        .max(0);
    if atom_count <= 0
        || logits.size().len() != 2
        || logits.size().first().copied().unwrap_or(-1) != atom_count
    {
        return previous_atom_types.shallow_clone();
    }
    let probabilities = logits.softmax(-1, Kind::Float);
    let (confidence, predicted) = probabilities.max_dim(-1, false);
    let mut committed = Vec::with_capacity(atom_count as usize);
    for index in 0..atom_count {
        let previous_token = previous_atom_types.int64_value(&[index]);
        let predicted_token = predicted.int64_value(&[index]);
        let threshold = atom_type_commit_threshold(previous_token, predicted_token);
        if confidence.double_value(&[index]) >= threshold {
            committed.push(predicted_token);
        } else {
            committed.push(previous_token);
        }
    }
    Tensor::from_slice(&committed).to_device(logits.device())
}

fn atom_type_commit_threshold(previous_token: i64, predicted_token: i64) -> f64 {
    match (
        previous_token == HYDROGEN_TOKEN,
        predicted_token == HYDROGEN_TOKEN,
    ) {
        (false, true) => HEAVY_TO_HYDROGEN_COMMIT_CONFIDENCE_THRESHOLD,
        (true, false) => HYDROGEN_TO_HEAVY_COMMIT_CONFIDENCE_THRESHOLD,
        _ => BASE_ATOM_TYPE_COMMIT_CONFIDENCE_THRESHOLD,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Device, Kind};

    #[test]
    fn confidence_gated_atom_type_commit_preserves_low_confidence_tokens() {
        let previous = Tensor::from_slice(&[1_i64, 2]);
        let logits = Tensor::zeros([2, 4], (Kind::Float, Device::Cpu));

        let committed = confidence_gated_atom_type_commit(&logits, &previous);

        assert_eq!(committed.int64_value(&[0]), 1);
        assert_eq!(committed.int64_value(&[1]), 2);
    }

    #[test]
    fn confidence_gated_atom_type_commit_updates_high_confidence_tokens() {
        let previous = Tensor::from_slice(&[0_i64, 0]);
        let logits =
            Tensor::from_slice(&[0.0_f32, 5.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0]).reshape([2, 4]);

        let committed = confidence_gated_atom_type_commit(&logits, &previous);

        assert_eq!(committed.int64_value(&[0]), 1);
        assert_eq!(committed.int64_value(&[1]), 2);
    }

    #[test]
    fn confidence_gated_atom_type_commit_requires_high_confidence_for_heavy_to_hydrogen() {
        let previous = Tensor::from_slice(&[0_i64]);
        let logits = Tensor::from_slice(&[0.0_f32, 0.0, 0.0, 0.0, 4.0]).reshape([1, 5]);

        let committed = confidence_gated_atom_type_commit(&logits, &previous);

        assert_eq!(committed.int64_value(&[0]), 0);
    }

    #[test]
    fn confidence_gated_atom_type_commit_repairs_hydrogen_to_heavy_with_moderate_confidence() {
        let previous = Tensor::from_slice(&[4_i64]);
        let logits = Tensor::from_slice(&[2.4_f32, 0.0, 0.0, 0.0, 0.0]).reshape([1, 5]);

        let committed = confidence_gated_atom_type_commit(&logits, &previous);

        assert_eq!(committed.int64_value(&[0]), 0);
    }
}
