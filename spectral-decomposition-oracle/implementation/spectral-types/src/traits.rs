//! Core traits for the spectral decomposition oracle.
//!
//! Defines generic interfaces for matrices, vectors, decomposable structures,
//! feature extraction, classification, solving, serialization, and validation.

use crate::error;
use crate::scalar::Scalar;
use std::fmt;

/// Trait for matrix-like objects.
pub trait MatrixLike<T: Scalar>: fmt::Debug + Send + Sync {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;

    fn get(&self, row: usize, col: usize) -> Option<T>;

    fn shape(&self) -> (usize, usize) {
        (self.rows(), self.cols())
    }

    fn is_square(&self) -> bool {
        self.rows() == self.cols()
    }

    fn is_empty(&self) -> bool {
        self.rows() == 0 || self.cols() == 0
    }

    fn nnz(&self) -> usize;

    fn density(&self) -> f64 {
        let total = self.rows() * self.cols();
        if total == 0 {
            0.0
        } else {
            self.nnz() as f64 / total as f64
        }
    }

    fn trace(&self) -> T {
        let n = self.rows().min(self.cols());
        let mut sum = T::zero();
        for i in 0..n {
            if let Some(v) = self.get(i, i) {
                sum = sum + v;
            }
        }
        sum
    }

    fn frobenius_norm_sq(&self) -> T;

    fn frobenius_norm(&self) -> T {
        self.frobenius_norm_sq().sqrt()
    }

    fn mul_vec(&self, x: &[T], y: &mut [T]) -> error::Result<()>;

    fn is_symmetric(&self, tol: T) -> bool {
        if !self.is_square() {
            return false;
        }
        let n = self.rows();
        for i in 0..n {
            for j in (i + 1)..n {
                let aij = self.get(i, j).unwrap_or(T::zero());
                let aji = self.get(j, i).unwrap_or(T::zero());
                if !aij.approx_eq(aji, tol) {
                    return false;
                }
            }
        }
        true
    }
}

/// Trait for vector-like objects.
pub trait VectorLike<T: Scalar>: fmt::Debug + Send + Sync {
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn get(&self, index: usize) -> Option<T>;

    fn as_slice(&self) -> &[T];

    fn dot(&self, other: &dyn VectorLike<T>) -> T {
        let n = self.len().min(other.len());
        let mut sum = T::zero();
        for i in 0..n {
            let a = self.get(i).unwrap_or(T::zero());
            let b = other.get(i).unwrap_or(T::zero());
            sum = sum + a * b;
        }
        sum
    }

    fn norm_l2(&self) -> T where Self: Sized {
        self.dot(self).sqrt()
    }

    fn norm_l1(&self) -> T {
        let s = self.as_slice();
        s.iter().copied().map(|v| v.abs()).fold(T::zero(), |a, b| a + b)
    }

    fn norm_linf(&self) -> T {
        let s = self.as_slice();
        s.iter()
            .copied()
            .map(|v| v.abs())
            .fold(T::zero(), |a, b| a.ordered_max(b))
    }
}

/// Trait for objects that can be decomposed.
pub trait Decomposable: fmt::Debug + Send + Sync {
    type Partition;
    type Config;
    type Result;

    fn detect_structure(&self) -> error::Result<crate::decomposition::DetectionResult>;

    fn decompose(
        &self,
        partition: &Self::Partition,
        config: &Self::Config,
    ) -> error::Result<Self::Result>;
}

/// Trait for feature extraction from any source.
pub trait FeatureExtractor: fmt::Debug + Send + Sync {
    type Input;
    type Features;

    fn extract(&self, input: &Self::Input) -> error::Result<Self::Features>;

    fn feature_names(&self) -> Vec<String>;

    fn feature_count(&self) -> usize {
        self.feature_names().len()
    }
}

/// Trait for classifiers.
pub trait Classifier: fmt::Debug + Send + Sync {
    type Features;
    type Prediction;

    fn predict(&self, features: &Self::Features) -> error::Result<Self::Prediction>;

    fn predict_proba(&self, features: &Self::Features) -> error::Result<Vec<f64>>;

    fn class_count(&self) -> usize;

    fn class_names(&self) -> Vec<String>;
}

/// Trait for solvers.
pub trait Solver: fmt::Debug + Send + Sync {
    type Problem;
    type Solution;
    type Config;

    fn solve(
        &self,
        problem: &Self::Problem,
        config: &Self::Config,
    ) -> error::Result<Self::Solution>;

    fn name(&self) -> &str;

    fn supports(&self, problem: &Self::Problem) -> bool;
}

/// Trait for serializable objects.
pub trait Serializable: Sized {
    fn to_json(&self) -> error::Result<String>;
    fn from_json(json: &str) -> error::Result<Self>;
    fn to_bytes(&self) -> error::Result<Vec<u8>>;
    fn from_bytes(bytes: &[u8]) -> error::Result<Self>;
}

/// Blanket implementation for Serialize + DeserializeOwned types.
impl<T> Serializable for T
where
    T: serde::Serialize + serde::de::DeserializeOwned,
{
    fn to_json(&self) -> error::Result<String> {
        serde_json::to_string_pretty(self).map_err(|e| {
            error::SpectralError::Io(error::IoError::JsonError {
                reason: e.to_string(),
            })
        })
    }

    fn from_json(json: &str) -> error::Result<Self> {
        serde_json::from_str(json).map_err(|e| {
            error::SpectralError::Io(error::IoError::JsonError {
                reason: e.to_string(),
            })
        })
    }

    fn to_bytes(&self) -> error::Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(|e| {
            error::SpectralError::Io(error::IoError::JsonError {
                reason: e.to_string(),
            })
        })
    }

    fn from_bytes(bytes: &[u8]) -> error::Result<Self> {
        serde_json::from_slice(bytes).map_err(|e| {
            error::SpectralError::Io(error::IoError::JsonError {
                reason: e.to_string(),
            })
        })
    }
}

/// Trait for validatable objects.
pub trait Validatable {
    fn validate(&self) -> error::Result<()>;

    fn validate_collect(&self) -> error::ErrorCollector {
        let mut collector = error::ErrorCollector::new();
        if let Err(e) = self.validate() {
            collector.push(error::ErrorSeverity::Error, e);
        }
        collector
    }

    fn is_valid(&self) -> bool {
        self.validate().is_ok()
    }
}

/// Trait for objects with a sparsity pattern.
pub trait SparsePattern {
    fn nnz(&self) -> usize;
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;

    fn density(&self) -> f64 {
        let total = self.rows() * self.cols();
        if total == 0 {
            0.0
        } else {
            self.nnz() as f64 / total as f64
        }
    }

    fn is_sparse(&self) -> bool {
        self.density() < 0.1
    }
}

/// Trait for iterating over non-zero entries.
pub trait SparseIterator<T: Scalar> {
    fn iter_nonzeros(&self) -> Box<dyn Iterator<Item = (usize, usize, T)> + '_>;
}

/// Trait for objects that can report their memory usage.
pub trait MemoryUsage {
    fn memory_bytes(&self) -> usize;

    fn memory_kb(&self) -> f64 {
        self.memory_bytes() as f64 / 1024.0
    }

    fn memory_mb(&self) -> f64 {
        self.memory_bytes() as f64 / (1024.0 * 1024.0)
    }
}

/// Trait for objects that can be scaled.
pub trait Scalable<T: Scalar> {
    fn scale(&mut self, factor: T);
    fn scaled(&self, factor: T) -> Self
    where
        Self: Clone,
    {
        let mut c = self.clone();
        c.scale(factor);
        c
    }
}

/// Trait for objects that support transposition.
pub trait Transposable {
    type Output;
    fn transpose(&self) -> Self::Output;
}

/// Trait for named entities.
pub trait Named {
    fn name(&self) -> &str;
    fn description(&self) -> &str {
        ""
    }
}

/// Progress callback for long-running operations.
pub type ProgressCallback = Box<dyn Fn(f64, &str) + Send + Sync>;

/// Trait for cancellable operations.
pub trait Cancellable {
    fn is_cancelled(&self) -> bool;
    fn cancel(&self);
}

/// Simple cancellation token.
#[derive(Debug)]
pub struct CancellationToken {
    cancelled: std::sync::atomic::AtomicBool,
}

impl CancellationToken {
    pub fn new() -> Self {
        Self {
            cancelled: std::sync::atomic::AtomicBool::new(false),
        }
    }
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

impl Cancellable for CancellationToken {
    fn is_cancelled(&self) -> bool {
        self.cancelled.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn cancel(&self) {
        self.cancelled
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct TestVec(Vec<f64>);

    impl VectorLike<f64> for TestVec {
        fn len(&self) -> usize {
            self.0.len()
        }
        fn get(&self, i: usize) -> Option<f64> {
            self.0.get(i).copied()
        }
        fn as_slice(&self) -> &[f64] {
            &self.0
        }
    }

    #[test]
    fn test_vector_dot() {
        let a = TestVec(vec![1.0, 2.0, 3.0]);
        let b = TestVec(vec![4.0, 5.0, 6.0]);
        let d = a.dot(&b);
        assert!((d - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector_norms() {
        let v = TestVec(vec![3.0, -4.0]);
        assert!((v.norm_l2() - 5.0).abs() < 1e-10);
        assert!((v.norm_l1() - 7.0).abs() < 1e-10);
        assert!((v.norm_linf() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector_empty() {
        let v = TestVec(vec![]);
        assert!(v.is_empty());
        assert_eq!(v.len(), 0);
    }

    #[test]
    fn test_serializable_roundtrip() {
        let data = vec![1.0_f64, 2.0, 3.0];
        let json = data.to_json().unwrap();
        let back: Vec<f64> = Serializable::from_json(&json).unwrap();
        assert_eq!(data, back);
    }

    #[test]
    fn test_serializable_bytes() {
        let val = 42_u32;
        let bytes = val.to_bytes().unwrap();
        let back: u32 = Serializable::from_bytes(&bytes).unwrap();
        assert_eq!(val, back);
    }

    #[test]
    fn test_cancellation_token() {
        let token = CancellationToken::new();
        assert!(!token.is_cancelled());
        token.cancel();
        assert!(token.is_cancelled());
    }

    #[test]
    fn test_sparse_pattern() {
        struct TestPattern;
        impl SparsePattern for TestPattern {
            fn nnz(&self) -> usize { 10 }
            fn rows(&self) -> usize { 100 }
            fn cols(&self) -> usize { 100 }
        }
        let p = TestPattern;
        assert!((p.density() - 0.001).abs() < 1e-10);
        assert!(p.is_sparse());
    }

    #[test]
    fn test_memory_usage() {
        struct TestMem;
        impl MemoryUsage for TestMem {
            fn memory_bytes(&self) -> usize { 1024 * 1024 }
        }
        let m = TestMem;
        assert!((m.memory_kb() - 1024.0).abs() < 1e-10);
        assert!((m.memory_mb() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_validatable() {
        #[derive(Debug)]
        struct AlwaysValid;
        impl Validatable for AlwaysValid {
            fn validate(&self) -> error::Result<()> { Ok(()) }
        }
        assert!(AlwaysValid.is_valid());
        assert!(AlwaysValid.validate_collect().is_empty());
    }

    #[test]
    fn test_validatable_invalid() {
        #[derive(Debug)]
        struct AlwaysInvalid;
        impl Validatable for AlwaysInvalid {
            fn validate(&self) -> error::Result<()> {
                Err(error::SpectralError::Internal("bad".into()))
            }
        }
        assert!(!AlwaysInvalid.is_valid());
        assert!(AlwaysInvalid.validate_collect().has_errors());
    }

    #[test]
    fn test_named() {
        struct TestNamed;
        impl Named for TestNamed {
            fn name(&self) -> &str { "test" }
        }
        assert_eq!(TestNamed.name(), "test");
        assert_eq!(TestNamed.description(), "");
    }

    #[test]
    fn test_cancellation_default() {
        let token = CancellationToken::default();
        assert!(!token.is_cancelled());
    }
}
