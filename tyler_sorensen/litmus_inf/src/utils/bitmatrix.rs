//! Utility bit matrix operations for LITMUS∞.
//!
//! Provides a lightweight bit matrix wrapper with utility operations
//! complementing the core BitMatrix in checker::execution.

/// A utility-level bit matrix for auxiliary computations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UtilBitMatrix {
    n: usize,
    data: Vec<u64>,
}

impl UtilBitMatrix {
    /// Create a zero-filled n×n bit matrix.
    pub fn new(n: usize) -> Self {
        let words_per_row = (n + 63) / 64;
        Self {
            n,
            data: vec![0u64; n * words_per_row],
        }
    }

    fn words_per_row(&self) -> usize {
        (self.n + 63) / 64
    }

    /// Get bit (i,j).
    pub fn get(&self, i: usize, j: usize) -> bool {
        let wpr = self.words_per_row();
        let word = j / 64;
        let bit = j % 64;
        (self.data[i * wpr + word] >> bit) & 1 == 1
    }

    /// Set bit (i,j).
    pub fn set(&mut self, i: usize, j: usize, val: bool) {
        let wpr = self.words_per_row();
        let word = j / 64;
        let bit = j % 64;
        if val {
            self.data[i * wpr + word] |= 1u64 << bit;
        } else {
            self.data[i * wpr + word] &= !(1u64 << bit);
        }
    }

    /// Dimension.
    pub fn dim(&self) -> usize {
        self.n
    }

    /// Count set bits.
    pub fn popcount(&self) -> usize {
        self.data.iter().map(|w| w.count_ones() as usize).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let mut m = UtilBitMatrix::new(4);
        m.set(0, 1, true);
        m.set(2, 3, true);
        assert!(m.get(0, 1));
        assert!(m.get(2, 3));
        assert!(!m.get(1, 0));
        assert_eq!(m.popcount(), 2);
    }
}
