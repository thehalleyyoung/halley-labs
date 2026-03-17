use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::{BitAnd, BitOr, BitXor, Not};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BitWidth { W8, W16, W32, W64, W128, W256 }

impl BitWidth {
    pub fn bits(self) -> u32 {
        match self { Self::W8=>8, Self::W16=>16, Self::W32=>32, Self::W64=>64, Self::W128=>128, Self::W256=>256 }
    }
    pub fn bytes(self) -> u32 { self.bits() / 8 }
    pub fn mask(self) -> u64 {
        match self { Self::W64 | Self::W128 | Self::W256 => u64::MAX, _ => (1u64 << self.bits()) - 1 }
    }
    pub fn from_bits(b: u32) -> Option<Self> {
        match b { 8=>Some(Self::W8), 16=>Some(Self::W16), 32=>Some(Self::W32), 64=>Some(Self::W64), 128=>Some(Self::W128), 256=>Some(Self::W256), _=>None }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BitVector { pub width: BitWidth, pub value: u64, pub high: u64 }

impl BitVector {
    pub fn new(width: BitWidth, value: u64) -> Self { Self { width, value: value & width.mask(), high: 0 } }
    pub fn zero(width: BitWidth) -> Self { Self::new(width, 0) }
    pub fn ones(width: BitWidth) -> Self { Self::new(width, width.mask()) }
    pub fn from_i64(width: BitWidth, v: i64) -> Self { Self::new(width, v as u64) }
    pub fn as_u64(&self) -> u64 { self.value }
    pub fn as_i64(&self) -> i64 {
        let shift = 64u32.saturating_sub(self.width.bits());
        ((self.value << shift) as i64) >> shift
    }
    pub fn bit(&self, index: u32) -> bool {
        if index < 64 { (self.value >> index) & 1 == 1 }
        else if index < 128 { (self.high >> (index - 64)) & 1 == 1 }
        else { false }
    }
    pub fn set_bit(&mut self, index: u32, val: bool) {
        if index < 64 {
            if val { self.value |= 1 << index; } else { self.value &= !(1 << index); }
        } else if index < 128 {
            let i = index - 64;
            if val { self.high |= 1 << i; } else { self.high &= !(1 << i); }
        }
    }
    pub fn popcount(&self) -> u32 { self.value.count_ones() + self.high.count_ones() }
    pub fn leading_zeros(&self) -> u32 {
        if self.width.bits() <= 64 { self.value.leading_zeros().saturating_sub(64 - self.width.bits()) }
        else { self.high.leading_zeros() }
    }
    pub fn trailing_zeros(&self) -> u32 {
        if self.value != 0 { self.value.trailing_zeros() } else { 64 + self.high.trailing_zeros() }
    }
    pub fn add(&self, other: &Self) -> Self { Self::new(self.width, self.value.wrapping_add(other.value)) }
    pub fn sub(&self, other: &Self) -> Self { Self::new(self.width, self.value.wrapping_sub(other.value)) }
    pub fn mul(&self, other: &Self) -> Self { Self::new(self.width, self.value.wrapping_mul(other.value)) }
    pub fn is_zero(&self) -> bool { self.value == 0 && self.high == 0 }
    pub fn is_negative(&self) -> bool { self.as_i64() < 0 }
    pub fn sign_extend(&self, to: BitWidth) -> Self { Self::new(to, self.as_i64() as u64) }
    pub fn zero_extend(&self, to: BitWidth) -> Self { Self::new(to, self.value) }
    pub fn truncate(&self, to: BitWidth) -> Self { Self::new(to, self.value) }
}

impl BitAnd for BitVector {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self { Self { width: self.width, value: self.value & rhs.value, high: self.high & rhs.high } }
}
impl BitOr for BitVector {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self { Self { width: self.width, value: self.value | rhs.value, high: self.high | rhs.high } }
}
impl BitXor for BitVector {
    type Output = Self;
    fn bitxor(self, rhs: Self) -> Self { Self { width: self.width, value: self.value ^ rhs.value, high: self.high ^ rhs.high } }
}
impl Not for BitVector {
    type Output = Self;
    fn not(self) -> Self { Self::new(self.width, !self.value) }
}
impl fmt::Display for BitVector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "bv{}(0x{:x})", self.width.bits(), self.value) }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn test_bv_basic() { let b = BitVector::new(BitWidth::W8, 0xFF); assert_eq!(b.as_u64(), 0xFF); assert_eq!(b.as_i64(), -1); }
    #[test] fn test_bv_ops() { let a = BitVector::new(BitWidth::W32, 0xF0); let b = BitVector::new(BitWidth::W32, 0x0F);
        assert_eq!((a.clone() | b.clone()).as_u64(), 0xFF); assert_eq!((a & b).as_u64(), 0x00); }
    #[test] fn test_bv_arith() { let a = BitVector::new(BitWidth::W8, 200); let b = BitVector::new(BitWidth::W8, 100);
        assert_eq!(a.add(&b).as_u64(), 44); }
}
