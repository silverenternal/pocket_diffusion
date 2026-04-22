//! Runtime helpers shared by config-driven training and experiment paths.

use std::error::Error;
use std::fmt;

use tch::Device;

use crate::config::RuntimeConfig;

/// Runtime-device parse and availability error.
#[derive(Debug, Clone)]
pub struct DeviceParseError {
    message: String,
}

impl DeviceParseError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for DeviceParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl Error for DeviceParseError {}

/// Parse a configured runtime device string into a `tch::Device`.
pub fn parse_runtime_device(device: &str) -> Result<Device, DeviceParseError> {
    let trimmed = device.trim().to_ascii_lowercase();
    if trimmed == "cpu" {
        return Ok(Device::Cpu);
    }

    if let Some(index) = trimmed.strip_prefix("cuda:") {
        let index = index.parse::<usize>().map_err(|_| {
            DeviceParseError::new(format!("invalid CUDA device index in `{device}`"))
        })?;
        return resolve_cuda_device(index, device);
    }

    if trimmed == "cuda" {
        return resolve_cuda_device(0, device);
    }

    Err(DeviceParseError::new(format!(
        "unsupported runtime device `{device}`; use `cpu`, `cuda`, or `cuda:N`"
    )))
}

fn resolve_cuda_device(index: usize, raw: &str) -> Result<Device, DeviceParseError> {
    if !tch::Cuda::is_available() {
        return Err(DeviceParseError::new(format!(
            "requested `{raw}` but CUDA is not available in this runtime"
        )));
    }
    let count = tch::Cuda::device_count();
    if index >= count as usize {
        return Err(DeviceParseError::new(format!(
            "requested `{raw}` but only {count} CUDA device(s) are available"
        )));
    }
    Ok(Device::Cuda(index))
}

impl RuntimeConfig {
    /// Resolve the configured device string into a concrete runtime device.
    pub fn resolve_device(&self) -> Result<Device, DeviceParseError> {
        parse_runtime_device(&self.device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_cpu_device() {
        assert_eq!(parse_runtime_device("cpu").unwrap(), Device::Cpu);
    }

    #[test]
    fn reject_invalid_device_string() {
        let err = parse_runtime_device("gpu0").unwrap_err();
        assert!(err
            .to_string()
            .contains("unsupported runtime device `gpu0`"));
    }
}
