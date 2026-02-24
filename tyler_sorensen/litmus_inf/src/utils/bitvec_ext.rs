//! Extended bitvector operations for LITMUS∞.
//!
//! Provides ExtBitVec with arithmetic, comparison, pattern matching,
//! population count variants, set operations, and encoding utilities.

use std::fmt;
use std::ops::{BitAnd, BitOr, BitXor, Not, Shl, Shr, BitAndAssign, BitOrAssign, BitXorAssign};
use std::cmp::Ordering;

// ---------------------------------------------------------------------------
// BitWidth
// ---------------------------------------------------------------------------

/// Supported bit widths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BitWidth {
    B4,
    B8,
    B16,
    B32,
    B64,
    B128,
    B256,
    Custom(usize),
}

impl BitWidth {
    /// Number of bits.
    pub fn bits(&self) -> usize {
        match self {
            Self::B4 => 4,
            Self::B8 => 8,
            Self::B16 => 16,
            Self::B32 => 32,
            Self::B64 => 64,
            Self::B128 => 128,
            Self::B256 => 256,
            Self::Custom(n) => *n,
        }
    }

    /// Number of u64 words needed.
    pub fn words(&self) -> usize {
        (self.bits() + 63) / 64
    }
}

impl fmt::Display for BitWidth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}bit", self.bits())
    }
}

// ---------------------------------------------------------------------------
// ExtBitVec
// ---------------------------------------------------------------------------

/// Extended bitvector with word-level operations.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct ExtBitVec {
    /// Backing store (least significant word first).
    words: Vec<u64>,
    /// Width in bits.
    width: usize,
}

impl ExtBitVec {
    /// Create a zero-filled bitvector of the given width.
    pub fn new(width: BitWidth) -> Self {
        let bits = width.bits();
        let num_words = width.words();
        Self {
            words: vec![0u64; num_words],
            width: bits,
        }
    }

    /// Create from a u64 value.
    pub fn from_u64(value: u64) -> Self {
        Self {
            words: vec![value],
            width: 64,
        }
    }

    /// Create from a u64 value with specified width.
    pub fn from_u64_with_width(value: u64, width: BitWidth) -> Self {
        let bits = width.bits();
        let num_words = width.words();
        let mut words = vec![0u64; num_words];
        words[0] = value;
        let mut bv = Self { words, width: bits };
        bv.mask_unused_bits();
        bv
    }

    /// Create from bytes (least significant byte first).
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let width = bytes.len() * 8;
        let num_words = (width + 63) / 64;
        let mut words = vec![0u64; num_words];
        for (i, &b) in bytes.iter().enumerate() {
            let word_idx = i / 8;
            let byte_idx = i % 8;
            words[word_idx] |= (b as u64) << (byte_idx * 8);
        }
        Self { words, width }
    }

    /// Create an all-zeros bitvector.
    pub fn zeros(width: BitWidth) -> Self {
        Self::new(width)
    }

    /// Create an all-ones bitvector.
    pub fn ones(width: BitWidth) -> Self {
        let bits = width.bits();
        let num_words = width.words();
        let mut words = vec![u64::MAX; num_words];
        // Mask the top word.
        let top_bits = bits % 64;
        if top_bits > 0 && !words.is_empty() {
            let last = words.len() - 1;
            words[last] = (1u64 << top_bits) - 1;
        }
        Self { words, width: bits }
    }

    /// Create from a binary string ("10110").
    pub fn from_binary_string(s: &str) -> Self {
        let width = s.len();
        let num_words = (width + 63) / 64;
        let mut words = vec![0u64; num_words];
        // MSB first in string.
        for (i, ch) in s.chars().rev().enumerate() {
            if ch == '1' {
                let word_idx = i / 64;
                let bit_idx = i % 64;
                words[word_idx] |= 1u64 << bit_idx;
            }
        }
        Self { words, width }
    }

    /// Width in bits.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Number of backing words.
    pub fn word_count(&self) -> usize {
        self.words.len()
    }

    /// Get the backing words.
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    /// Convert to u64 (truncates if wider than 64 bits).
    pub fn to_u64(&self) -> u64 {
        self.words.first().copied().unwrap_or(0)
    }

    /// Mask unused bits in the top word.
    fn mask_unused_bits(&mut self) {
        let top_bits = self.width % 64;
        if top_bits > 0 && !self.words.is_empty() {
            let last = self.words.len() - 1;
            self.words[last] &= (1u64 << top_bits) - 1;
        }
    }

    // ---- Bit manipulation ----

    /// Get bit at position.
    pub fn get_bit(&self, pos: usize) -> bool {
        if pos >= self.width { return false; }
        let word_idx = pos / 64;
        let bit_idx = pos % 64;
        (self.words[word_idx] >> bit_idx) & 1 == 1
    }

    /// Set bit at position.
    pub fn set_bit(&mut self, pos: usize, val: bool) {
        if pos >= self.width { return; }
        let word_idx = pos / 64;
        let bit_idx = pos % 64;
        if val {
            self.words[word_idx] |= 1u64 << bit_idx;
        } else {
            self.words[word_idx] &= !(1u64 << bit_idx);
        }
    }

    /// Clear a bit.
    pub fn clear_bit(&mut self, pos: usize) {
        self.set_bit(pos, false);
    }

    /// Toggle a bit.
    pub fn toggle_bit(&mut self, pos: usize) {
        if pos >= self.width { return; }
        let word_idx = pos / 64;
        let bit_idx = pos % 64;
        self.words[word_idx] ^= 1u64 << bit_idx;
    }

    /// Extract bits from range [lo, hi) as a new bitvector.
    pub fn bit_range(&self, lo: usize, hi: usize) -> Self {
        let width = hi.saturating_sub(lo);
        let bw = BitWidth::Custom(width);
        let mut result = Self::new(bw);
        for i in 0..width {
            result.set_bit(i, self.get_bit(lo + i));
        }
        result
    }

    // ---- Population count ----

    /// Total number of set bits.
    pub fn popcount(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Count set bits in range [start, end).
    pub fn partial_popcount(&self, start: usize, end: usize) -> usize {
        let mut count = 0;
        for i in start..end.min(self.width) {
            if self.get_bit(i) { count += 1; }
        }
        count
    }

    /// Number of leading zeros (from MSB).
    pub fn leading_zeros(&self) -> usize {
        let mut count = 0;
        for i in (0..self.width).rev() {
            if self.get_bit(i) { break; }
            count += 1;
        }
        count
    }

    /// Number of trailing zeros (from LSB).
    pub fn trailing_zeros(&self) -> usize {
        if self.words.is_empty() { return self.width; }
        let mut count = 0;
        for &w in &self.words {
            if w == 0 {
                count += 64;
            } else {
                count += w.trailing_zeros() as usize;
                break;
            }
        }
        count.min(self.width)
    }

    /// Number of leading ones (from MSB).
    pub fn leading_ones(&self) -> usize {
        let mut count = 0;
        for i in (0..self.width).rev() {
            if !self.get_bit(i) { break; }
            count += 1;
        }
        count
    }

    /// Number of trailing ones (from LSB).
    pub fn trailing_ones(&self) -> usize {
        let mut count = 0;
        for i in 0..self.width {
            if !self.get_bit(i) { break; }
            count += 1;
        }
        count
    }

    /// Check if all bits are zero.
    pub fn is_zero(&self) -> bool {
        self.words.iter().all(|&w| w == 0)
    }

    /// Check if all bits are one (within width).
    pub fn is_all_ones(&self) -> bool {
        self.popcount() == self.width
    }

    // ---- Arithmetic ----

    /// Add two bitvectors. Returns (result, carry).
    pub fn add(&self, other: &Self) -> (Self, bool) {
        let width = self.width.max(other.width);
        let num_words = (width + 63) / 64;
        let mut result = vec![0u64; num_words];
        let mut carry = 0u64;

        for i in 0..num_words {
            let a = self.words.get(i).copied().unwrap_or(0);
            let b = other.words.get(i).copied().unwrap_or(0);
            let (sum1, c1) = a.overflowing_add(b);
            let (sum2, c2) = sum1.overflowing_add(carry);
            result[i] = sum2;
            carry = (c1 as u64) + (c2 as u64);
        }

        let mut bv = Self { words: result, width };
        bv.mask_unused_bits();
        (bv, carry > 0)
    }

    /// Subtract other from self. Returns (result, borrow).
    pub fn sub(&self, other: &Self) -> (Self, bool) {
        let width = self.width.max(other.width);
        let num_words = (width + 63) / 64;
        let mut result = vec![0u64; num_words];
        let mut borrow = 0u64;

        for i in 0..num_words {
            let a = self.words.get(i).copied().unwrap_or(0);
            let b = other.words.get(i).copied().unwrap_or(0);
            let (diff1, b1) = a.overflowing_sub(b);
            let (diff2, b2) = diff1.overflowing_sub(borrow);
            result[i] = diff2;
            borrow = (b1 as u64) + (b2 as u64);
        }

        let mut bv = Self { words: result, width };
        bv.mask_unused_bits();
        (bv, borrow > 0)
    }

    /// Multiply two bitvectors (schoolbook method, result width = sum of widths).
    pub fn mul(&self, other: &Self) -> Self {
        let result_width = self.width + other.width;
        let num_words = (result_width + 63) / 64;
        let mut result = vec![0u64; num_words];

        for i in 0..self.words.len() {
            let mut carry = 0u128;
            for j in 0..other.words.len() {
                if i + j >= num_words { break; }
                let prod = (self.words[i] as u128) * (other.words[j] as u128)
                    + (result[i + j] as u128)
                    + carry;
                result[i + j] = prod as u64;
                carry = prod >> 64;
            }
            if i + other.words.len() < num_words {
                result[i + other.words.len()] = carry as u64;
            }
        }

        let mut bv = Self { words: result, width: result_width };
        bv.mask_unused_bits();
        bv
    }

    // ---- Comparison ----

    /// Unsigned comparison.
    pub fn unsigned_cmp(&self, other: &Self) -> Ordering {
        let max_words = self.words.len().max(other.words.len());
        for i in (0..max_words).rev() {
            let a = self.words.get(i).copied().unwrap_or(0);
            let b = other.words.get(i).copied().unwrap_or(0);
            match a.cmp(&b) {
                Ordering::Equal => continue,
                ord => return ord,
            }
        }
        Ordering::Equal
    }

    /// Signed comparison (MSB is sign bit).
    pub fn signed_cmp(&self, other: &Self) -> Ordering {
        let a_sign = self.get_bit(self.width.saturating_sub(1));
        let b_sign = other.get_bit(other.width.saturating_sub(1));

        match (a_sign, b_sign) {
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            _ => self.unsigned_cmp(other),
        }
    }

    /// Hamming distance between two bitvectors.
    pub fn hamming_distance(&self, other: &Self) -> usize {
        let max_words = self.words.len().max(other.words.len());
        let mut dist = 0;
        for i in 0..max_words {
            let a = self.words.get(i).copied().unwrap_or(0);
            let b = other.words.get(i).copied().unwrap_or(0);
            dist += (a ^ b).count_ones() as usize;
        }
        dist
    }

    /// Jaccard similarity (|A ∩ B| / |A ∪ B|).
    pub fn jaccard_similarity(&self, other: &Self) -> f64 {
        let max_words = self.words.len().max(other.words.len());
        let mut inter = 0usize;
        let mut union_count = 0usize;

        for i in 0..max_words {
            let a = self.words.get(i).copied().unwrap_or(0);
            let b = other.words.get(i).copied().unwrap_or(0);
            inter += (a & b).count_ones() as usize;
            union_count += (a | b).count_ones() as usize;
        }

        if union_count == 0 { 1.0 } else { inter as f64 / union_count as f64 }
    }

    // ---- Set operations ----

    /// Bitwise union (OR).
    pub fn union(&self, other: &Self) -> Self {
        let width = self.width.max(other.width);
        let num_words = (width + 63) / 64;
        let mut words = vec![0u64; num_words];
        for i in 0..num_words {
            let a = self.words.get(i).copied().unwrap_or(0);
            let b = other.words.get(i).copied().unwrap_or(0);
            words[i] = a | b;
        }
        let mut bv = Self { words, width };
        bv.mask_unused_bits();
        bv
    }

    /// Bitwise intersection (AND).
    pub fn intersection(&self, other: &Self) -> Self {
        let width = self.width.max(other.width);
        let num_words = (width + 63) / 64;
        let mut words = vec![0u64; num_words];
        for i in 0..num_words {
            let a = self.words.get(i).copied().unwrap_or(0);
            let b = other.words.get(i).copied().unwrap_or(0);
            words[i] = a & b;
        }
        Self { words, width }
    }

    /// Bitwise difference (AND NOT).
    pub fn difference(&self, other: &Self) -> Self {
        let width = self.width.max(other.width);
        let num_words = (width + 63) / 64;
        let mut words = vec![0u64; num_words];
        for i in 0..num_words {
            let a = self.words.get(i).copied().unwrap_or(0);
            let b = other.words.get(i).copied().unwrap_or(0);
            words[i] = a & !b;
        }
        let mut bv = Self { words, width };
        bv.mask_unused_bits();
        bv
    }

    /// Symmetric difference (XOR).
    pub fn symmetric_difference(&self, other: &Self) -> Self {
        let width = self.width.max(other.width);
        let num_words = (width + 63) / 64;
        let mut words = vec![0u64; num_words];
        for i in 0..num_words {
            let a = self.words.get(i).copied().unwrap_or(0);
            let b = other.words.get(i).copied().unwrap_or(0);
            words[i] = a ^ b;
        }
        let mut bv = Self { words, width };
        bv.mask_unused_bits();
        bv
    }

    /// Check if self is a subset of other.
    pub fn is_subset(&self, other: &Self) -> bool {
        self.intersection(other) == *self
    }

    /// Check if self is a superset of other.
    pub fn is_superset(&self, other: &Self) -> bool {
        other.is_subset(self)
    }

    // ---- Iteration ----

    /// Iterate over indices of set bits.
    pub fn iter_set_bits(&self) -> SetBitIterator<'_> {
        SetBitIterator {
            bv: self,
            pos: 0,
        }
    }

    /// Find the next set bit at or after `from`.
    pub fn next_set_bit(&self, from: usize) -> Option<usize> {
        for i in from..self.width {
            if self.get_bit(i) {
                return Some(i);
            }
        }
        None
    }

    /// Find the previous set bit at or before `from`.
    pub fn prev_set_bit(&self, from: usize) -> Option<usize> {
        let start = from.min(self.width.saturating_sub(1));
        for i in (0..=start).rev() {
            if self.get_bit(i) {
                return Some(i);
            }
        }
        None
    }

    // ---- Shifting ----

    /// Shift left by n bits.
    pub fn shift_left(&self, n: usize) -> Self {
        if n >= self.width {
            return Self::zeros(BitWidth::Custom(self.width));
        }
        let mut result = Self::zeros(BitWidth::Custom(self.width));
        for i in n..self.width {
            result.set_bit(i, self.get_bit(i - n));
        }
        result
    }

    /// Shift right by n bits.
    pub fn shift_right(&self, n: usize) -> Self {
        if n >= self.width {
            return Self::zeros(BitWidth::Custom(self.width));
        }
        let mut result = Self::zeros(BitWidth::Custom(self.width));
        for i in 0..(self.width - n) {
            result.set_bit(i, self.get_bit(i + n));
        }
        result
    }

    // ---- Encoding ----

    /// Convert to Gray code.
    pub fn to_gray_code(&self) -> Self {
        let shifted = self.shift_right(1);
        self.symmetric_difference(&shifted)
    }

    /// Convert from Gray code back to binary.
    pub fn from_gray_code(&self) -> Self {
        let mut result = self.clone();
        let mut shift = 1;
        while shift < self.width {
            let shifted = result.shift_right(shift);
            result = result.symmetric_difference(&shifted);
            shift *= 2;
        }
        result
    }

    /// Run-length encode: returns (value, length) pairs.
    pub fn run_length_encode(&self) -> Vec<(bool, usize)> {
        if self.width == 0 { return Vec::new(); }

        let mut runs = Vec::new();
        let mut current_val = self.get_bit(0);
        let mut current_len = 1;

        for i in 1..self.width {
            let bit = self.get_bit(i);
            if bit == current_val {
                current_len += 1;
            } else {
                runs.push((current_val, current_len));
                current_val = bit;
                current_len = 1;
            }
        }
        runs.push((current_val, current_len));
        runs
    }

    /// Decode from run-length encoding.
    pub fn from_run_length(runs: &[(bool, usize)]) -> Self {
        let total_bits: usize = runs.iter().map(|(_, len)| *len).sum();
        let mut bv = Self::zeros(BitWidth::Custom(total_bits));
        let mut pos = 0;
        for &(val, len) in runs {
            for i in 0..len {
                bv.set_bit(pos + i, val);
            }
            pos += len;
        }
        bv
    }

    /// Convert to binary string (MSB first).
    pub fn to_binary_string(&self) -> String {
        let mut s = String::with_capacity(self.width);
        for i in (0..self.width).rev() {
            s.push(if self.get_bit(i) { '1' } else { '0' });
        }
        s
    }

    /// Convert to hex string.
    pub fn to_hex_string(&self) -> String {
        let mut s = String::new();
        for &w in self.words.iter().rev() {
            if s.is_empty() {
                s.push_str(&format!("{:x}", w));
            } else {
                s.push_str(&format!("{:016x}", w));
            }
        }
        if s.is_empty() { s.push('0'); }
        s
    }
}

// ---- Iterator for set bits ----

/// Iterator over set bit positions.
pub struct SetBitIterator<'a> {
    bv: &'a ExtBitVec,
    pos: usize,
}

impl<'a> Iterator for SetBitIterator<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        while self.pos < self.bv.width {
            let pos = self.pos;
            self.pos += 1;
            if self.bv.get_bit(pos) {
                return Some(pos);
            }
        }
        None
    }
}

// ---- Pattern Matching ----

/// Pattern matcher for bitvectors.
#[derive(Debug, Clone)]
pub struct PatternMatcher;

impl PatternMatcher {
    /// Find positions where `needle` appears in `haystack`.
    pub fn find_pattern(needle: &ExtBitVec, haystack: &ExtBitVec) -> Vec<usize> {
        let mut positions = Vec::new();
        if needle.width() > haystack.width() {
            return positions;
        }
        for start in 0..=(haystack.width() - needle.width()) {
            let mut matches = true;
            for i in 0..needle.width() {
                if needle.get_bit(i) != haystack.get_bit(start + i) {
                    matches = false;
                    break;
                }
            }
            if matches {
                positions.push(start);
            }
        }
        positions
    }

    /// Wildcard match: pattern has three states (0, 1, don't-care).
    /// `mask`: 1 means the bit matters, 0 means don't-care.
    /// `pattern`: the expected values for bits where mask is 1.
    pub fn mask_match(
        value: &ExtBitVec, pattern: &ExtBitVec, mask: &ExtBitVec,
    ) -> bool {
        let width = value.width().min(pattern.width()).min(mask.width());
        for i in 0..width {
            if mask.get_bit(i) {
                if value.get_bit(i) != pattern.get_bit(i) {
                    return false;
                }
            }
        }
        true
    }

    /// Find all positions where a masked pattern matches.
    pub fn find_masked_pattern(
        haystack: &ExtBitVec, pattern: &ExtBitVec, mask: &ExtBitVec,
    ) -> Vec<usize> {
        let pat_width = pattern.width().min(mask.width());
        if pat_width > haystack.width() {
            return Vec::new();
        }

        let mut positions = Vec::new();
        for start in 0..=(haystack.width() - pat_width) {
            let mut matches = true;
            for i in 0..pat_width {
                if mask.get_bit(i) {
                    if haystack.get_bit(start + i) != pattern.get_bit(i) {
                        matches = false;
                        break;
                    }
                }
            }
            if matches {
                positions.push(start);
            }
        }
        positions
    }
}

// ---- Trait implementations ----

impl fmt::Debug for ExtBitVec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ExtBitVec({}, 0x{})", self.width, self.to_hex_string())
    }
}

impl fmt::Display for ExtBitVec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_binary_string())
    }
}

impl BitAnd for ExtBitVec {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self { self.intersection(&rhs) }
}

impl BitOr for ExtBitVec {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self { self.union(&rhs) }
}

impl BitXor for ExtBitVec {
    type Output = Self;
    fn bitxor(self, rhs: Self) -> Self { self.symmetric_difference(&rhs) }
}

impl Not for ExtBitVec {
    type Output = Self;
    fn not(self) -> Self {
        let mut words: Vec<u64> = self.words.iter().map(|w| !w).collect();
        let mut bv = Self { words, width: self.width };
        bv.mask_unused_bits();
        bv
    }
}

impl BitAndAssign for ExtBitVec {
    fn bitand_assign(&mut self, rhs: Self) {
        *self = self.intersection(&rhs);
    }
}

impl BitOrAssign for ExtBitVec {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = self.union(&rhs);
    }
}

impl BitXorAssign for ExtBitVec {
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = self.symmetric_difference(&rhs);
    }
}

impl Shl<usize> for ExtBitVec {
    type Output = Self;
    fn shl(self, n: usize) -> Self { self.shift_left(n) }
}

impl Shr<usize> for ExtBitVec {
    type Output = Self;
    fn shr(self, n: usize) -> Self { self.shift_right(n) }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Construction tests --

    #[test]
    fn test_new_zeros() {
        let bv = ExtBitVec::new(BitWidth::B64);
        assert_eq!(bv.width(), 64);
        assert!(bv.is_zero());
        assert_eq!(bv.popcount(), 0);
    }

    #[test]
    fn test_from_u64() {
        let bv = ExtBitVec::from_u64(0xFF);
        assert_eq!(bv.to_u64(), 0xFF);
        assert_eq!(bv.popcount(), 8);
    }

    #[test]
    fn test_from_binary_string() {
        let bv = ExtBitVec::from_binary_string("10110");
        assert_eq!(bv.width(), 5);
        assert!(bv.get_bit(0));  // LSB
        assert!(bv.get_bit(1));
        assert!(!bv.get_bit(2));
        assert!(bv.get_bit(3));
        assert!(!bv.get_bit(4));
    }

    #[test]
    fn test_ones() {
        let bv = ExtBitVec::ones(BitWidth::B8);
        assert_eq!(bv.popcount(), 8);
        assert!(bv.is_all_ones());
    }

    #[test]
    fn test_from_bytes() {
        let bv = ExtBitVec::from_bytes(&[0xFF, 0x00]);
        assert_eq!(bv.width(), 16);
        assert_eq!(bv.partial_popcount(0, 8), 8);
        assert_eq!(bv.partial_popcount(8, 16), 0);
    }

    // -- Bit manipulation tests --

    #[test]
    fn test_get_set_bit() {
        let mut bv = ExtBitVec::new(BitWidth::B32);
        bv.set_bit(5, true);
        assert!(bv.get_bit(5));
        assert!(!bv.get_bit(4));
        bv.clear_bit(5);
        assert!(!bv.get_bit(5));
    }

    #[test]
    fn test_toggle_bit() {
        let mut bv = ExtBitVec::new(BitWidth::B32);
        bv.toggle_bit(3);
        assert!(bv.get_bit(3));
        bv.toggle_bit(3);
        assert!(!bv.get_bit(3));
    }

    #[test]
    fn test_bit_range() {
        let bv = ExtBitVec::from_u64(0b11010110);
        let sub = bv.bit_range(2, 6); // bits 2,3,4,5 = 0101
        assert_eq!(sub.width(), 4);
        assert!(sub.get_bit(0));  // bit 2 of original = 1
    }

    // -- Popcount tests --

    #[test]
    fn test_popcount() {
        let bv = ExtBitVec::from_u64(0b10110101);
        assert_eq!(bv.popcount(), 5);
    }

    #[test]
    fn test_partial_popcount() {
        let bv = ExtBitVec::from_u64(0b11111111);
        assert_eq!(bv.partial_popcount(0, 4), 4);
        assert_eq!(bv.partial_popcount(4, 8), 4);
    }

    #[test]
    fn test_leading_trailing_zeros() {
        let bv = ExtBitVec::from_u64_with_width(0b00001000, BitWidth::B8);
        assert_eq!(bv.trailing_zeros(), 3);
        assert_eq!(bv.leading_zeros(), 4);
    }

    #[test]
    fn test_leading_trailing_ones() {
        let bv = ExtBitVec::from_u64_with_width(0b11100111, BitWidth::B8);
        assert_eq!(bv.trailing_ones(), 3);
        assert_eq!(bv.leading_ones(), 3);
    }

    // -- Arithmetic tests --

    #[test]
    fn test_add() {
        let a = ExtBitVec::from_u64(100);
        let b = ExtBitVec::from_u64(200);
        let (result, carry) = a.add(&b);
        assert_eq!(result.to_u64(), 300);
        assert!(!carry);
    }

    #[test]
    fn test_add_carry() {
        let a = ExtBitVec::from_u64(u64::MAX);
        let b = ExtBitVec::from_u64(1);
        let (result, carry) = a.add(&b);
        assert_eq!(result.to_u64(), 0);
        assert!(carry);
    }

    #[test]
    fn test_sub() {
        let a = ExtBitVec::from_u64(300);
        let b = ExtBitVec::from_u64(100);
        let (result, borrow) = a.sub(&b);
        assert_eq!(result.to_u64(), 200);
        assert!(!borrow);
    }

    #[test]
    fn test_sub_borrow() {
        let a = ExtBitVec::from_u64(0);
        let b = ExtBitVec::from_u64(1);
        let (_, borrow) = a.sub(&b);
        assert!(borrow);
    }

    #[test]
    fn test_mul() {
        let a = ExtBitVec::from_u64(7);
        let b = ExtBitVec::from_u64(6);
        let result = a.mul(&b);
        assert_eq!(result.to_u64(), 42);
    }

    // -- Comparison tests --

    #[test]
    fn test_unsigned_cmp() {
        let a = ExtBitVec::from_u64(100);
        let b = ExtBitVec::from_u64(200);
        assert_eq!(a.unsigned_cmp(&b), Ordering::Less);
        assert_eq!(b.unsigned_cmp(&a), Ordering::Greater);
        assert_eq!(a.unsigned_cmp(&a), Ordering::Equal);
    }

    #[test]
    fn test_hamming_distance() {
        let a = ExtBitVec::from_u64(0b1010);
        let b = ExtBitVec::from_u64(0b1100);
        assert_eq!(a.hamming_distance(&b), 2);
    }

    #[test]
    fn test_jaccard_similarity() {
        let a = ExtBitVec::from_u64(0b1111);
        let b = ExtBitVec::from_u64(0b1111);
        assert!((a.jaccard_similarity(&b) - 1.0).abs() < 1e-10);

        let c = ExtBitVec::from_u64(0b0000);
        let d = ExtBitVec::from_u64(0b0000);
        assert!((c.jaccard_similarity(&d) - 1.0).abs() < 1e-10);
    }

    // -- Set operation tests --

    #[test]
    fn test_union() {
        let a = ExtBitVec::from_u64(0b1010);
        let b = ExtBitVec::from_u64(0b0110);
        let result = a.union(&b);
        assert_eq!(result.to_u64(), 0b1110);
    }

    #[test]
    fn test_intersection() {
        let a = ExtBitVec::from_u64(0b1010);
        let b = ExtBitVec::from_u64(0b0110);
        let result = a.intersection(&b);
        assert_eq!(result.to_u64(), 0b0010);
    }

    #[test]
    fn test_difference() {
        let a = ExtBitVec::from_u64(0b1010);
        let b = ExtBitVec::from_u64(0b0110);
        let result = a.difference(&b);
        assert_eq!(result.to_u64(), 0b1000);
    }

    #[test]
    fn test_symmetric_difference() {
        let a = ExtBitVec::from_u64(0b1010);
        let b = ExtBitVec::from_u64(0b0110);
        let result = a.symmetric_difference(&b);
        assert_eq!(result.to_u64(), 0b1100);
    }

    #[test]
    fn test_subset_superset() {
        let a = ExtBitVec::from_u64(0b0010);
        let b = ExtBitVec::from_u64(0b1010);
        assert!(a.is_subset(&b));
        assert!(b.is_superset(&a));
        assert!(!b.is_subset(&a));
    }

    // -- Shifting tests --

    #[test]
    fn test_shift_left() {
        let bv = ExtBitVec::from_u64_with_width(0b0001, BitWidth::B8);
        let shifted = bv.shift_left(2);
        assert_eq!(shifted.to_u64(), 0b0100);
    }

    #[test]
    fn test_shift_right() {
        let bv = ExtBitVec::from_u64_with_width(0b1000, BitWidth::B8);
        let shifted = bv.shift_right(2);
        assert_eq!(shifted.to_u64(), 0b0010);
    }

    // -- Encoding tests --

    #[test]
    fn test_gray_code_roundtrip() {
        let bv = ExtBitVec::from_u64_with_width(13, BitWidth::B8);
        let gray = bv.to_gray_code();
        let back = gray.from_gray_code();
        assert_eq!(bv.to_u64(), back.to_u64());
    }

    #[test]
    fn test_run_length_encoding() {
        let bv = ExtBitVec::from_binary_string("11100011");
        let runs = bv.run_length_encode();
        let decoded = ExtBitVec::from_run_length(&runs);
        assert_eq!(bv.to_binary_string(), decoded.to_binary_string());
    }

    // -- Pattern matching tests --

    #[test]
    fn test_find_pattern() {
        let haystack = ExtBitVec::from_binary_string("101101");
        let needle = ExtBitVec::from_binary_string("101");
        let positions = PatternMatcher::find_pattern(&needle, &haystack);
        assert!(!positions.is_empty());
    }

    #[test]
    fn test_mask_match() {
        let value   = ExtBitVec::from_binary_string("10110");
        let pattern = ExtBitVec::from_binary_string("10010");
        let mask    = ExtBitVec::from_binary_string("11011");
        // Bit 2 is don't-care (mask=0), others must match.
        assert!(PatternMatcher::mask_match(&value, &pattern, &mask));
    }

    // -- Iterator tests --

    #[test]
    fn test_iter_set_bits() {
        let bv = ExtBitVec::from_u64(0b10110);
        let bits: Vec<usize> = bv.iter_set_bits().collect();
        assert_eq!(bits, vec![1, 2, 4]);
    }

    #[test]
    fn test_next_set_bit() {
        let bv = ExtBitVec::from_u64(0b10100);
        assert_eq!(bv.next_set_bit(0), Some(2));
        assert_eq!(bv.next_set_bit(3), Some(4));
        assert_eq!(bv.next_set_bit(5), None);
    }

    #[test]
    fn test_prev_set_bit() {
        let bv = ExtBitVec::from_u64(0b10100);
        assert_eq!(bv.prev_set_bit(4), Some(4));
        assert_eq!(bv.prev_set_bit(3), Some(2));
        assert_eq!(bv.prev_set_bit(1), None);
    }

    // -- Operator trait tests --

    #[test]
    fn test_bitand_op() {
        let a = ExtBitVec::from_u64(0b1010);
        let b = ExtBitVec::from_u64(0b1100);
        let result = a & b;
        assert_eq!(result.to_u64(), 0b1000);
    }

    #[test]
    fn test_bitor_op() {
        let a = ExtBitVec::from_u64(0b1010);
        let b = ExtBitVec::from_u64(0b0101);
        let result = a | b;
        assert_eq!(result.to_u64(), 0b1111);
    }

    #[test]
    fn test_not_op() {
        let bv = ExtBitVec::from_u64_with_width(0b1010, BitWidth::B4);
        let result = !bv;
        assert_eq!(result.to_u64(), 0b0101);
    }

    // -- Display tests --

    #[test]
    fn test_binary_string() {
        let bv = ExtBitVec::from_u64_with_width(0b1010, BitWidth::B4);
        assert_eq!(bv.to_binary_string(), "1010");
    }

    #[test]
    fn test_hex_string() {
        let bv = ExtBitVec::from_u64(0xFF);
        assert_eq!(bv.to_hex_string(), "ff");
    }
}
