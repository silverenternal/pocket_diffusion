//! Integration tests for Mutual Information Monitor

use pocket_diffusion::losses::mutual_information::MutualInformationMonitor;
use tch::{Device, Kind, Tensor};

/// Helper: create random tensor on CPU
fn rand_tensor(shape: &[i64]) -> Tensor {
    Tensor::randn(shape, (Kind::Float, Device::Cpu))
}

#[test]
fn test_mi_identical_variables() {
    // MI(X; X) should be high (perfect correlation)
    let monitor = MutualInformationMonitor::new(32);
    let x = rand_tensor(&[256]);
    let mi = monitor.compute_mi(&x, &x);

    // MI should be significantly positive
    assert!(mi > 0.0, "MI(X;X) should be positive, got {}", mi);
}

#[test]
fn test_mi_independent_variables() {
    // MI(X; Y) for independent X,Y should be near zero
    let monitor = MutualInformationMonitor::new(32);
    let x = rand_tensor(&[512]);
    let y = rand_tensor(&[512]);
    let mi = monitor.compute_mi(&x, &y);

    // MI should be small (not exactly 0 due to estimation noise)
    assert!(mi >= 0.0, "MI should be non-negative, got {}", mi);
    assert!(
        mi < 1.0,
        "MI for independent vars should be small, got {}",
        mi
    );
}

#[test]
fn test_mi_empty_tensors() {
    let monitor = MutualInformationMonitor::new(32);
    let empty = Tensor::zeros([0], (Kind::Float, Device::Cpu));
    let mi = monitor.compute_mi(&empty, &empty);
    assert_eq!(mi, 0.0, "MI of empty tensors should be 0");
}

#[test]
fn test_mi_mismatched_sizes() {
    let monitor = MutualInformationMonitor::new(32);
    let x = rand_tensor(&[128]);
    let y = rand_tensor(&[256]);
    let mi = monitor.compute_mi(&x, &y);
    assert_eq!(mi, 0.0, "MI of mismatched sizes should return 0");
}

#[test]
fn test_mi_single_element() {
    let monitor = MutualInformationMonitor::new(32);
    let x = Tensor::ones([1], (Kind::Float, Device::Cpu));
    let y = Tensor::ones([1], (Kind::Float, Device::Cpu));
    let mi = monitor.compute_mi(&x, &y);
    assert_eq!(mi, 0.0, "MI of single element should be 0");
}

#[test]
fn test_mi_symmetric() {
    // MI(X; Y) == MI(Y; X)
    let monitor = MutualInformationMonitor::new(32);
    let x = rand_tensor(&[256]);
    let y = rand_tensor(&[256]);

    let mi_xy = monitor.compute_mi(&x, &y);
    let mi_yx = monitor.compute_mi(&y, &x);

    assert!(
        (mi_xy - mi_yx).abs() < 1e-6,
        "MI should be symmetric: MI(X;Y)={}, MI(Y;X)={}",
        mi_xy,
        mi_yx
    );
}

#[test]
fn test_decoupling_report() {
    use pocket_diffusion::losses::mutual_information::DecouplingQualityReport;

    let report = DecouplingQualityReport::new(0.1, 0.2, 0.3);
    assert_eq!(report.mi_topo_geo, 0.1);
    assert_eq!(report.mi_topo_pocket, 0.2);
    assert_eq!(report.mi_geo_pocket, 0.3);
    assert!((report.mi_mean - 0.2).abs() < 1e-6);
}
