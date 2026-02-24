//! Group representation theory for LITMUS∞ symmetry analysis.
//!
//! Implements matrix representations, characters, character tables,
//! irreducible decomposition, Fourier transforms on finite groups,
//! Burnside's theorem, Schur orthogonality, induced/restricted
//! representations, tensor products, and the representation ring.

use std::collections::{HashMap, HashSet, BTreeMap};
use std::fmt;
use std::f64::consts::PI;

use super::types::Permutation;
use super::group::PermutationGroup;

// ═══════════════════════════════════════════════════════════════════════
// Complex number type
// ═══════════════════════════════════════════════════════════════════════

/// A complex number with f64 real and imaginary parts.
#[derive(Clone, Copy)]
pub struct Complex {
    /// Real part.
    pub re: f64,
    /// Imaginary part.
    pub im: f64,
}

impl Complex {
    /// Create a new complex number.
    pub fn new(re: f64, im: f64) -> Self {
        Complex { re, im }
    }

    /// The zero complex number.
    pub fn zero() -> Self {
        Complex { re: 0.0, im: 0.0 }
    }

    /// The multiplicative identity.
    pub fn one() -> Self {
        Complex { re: 1.0, im: 0.0 }
    }

    /// Create a purely real complex number.
    pub fn from_real(re: f64) -> Self {
        Complex { re, im: 0.0 }
    }

    /// Create a purely imaginary complex number.
    pub fn from_imag(im: f64) -> Self {
        Complex { re: 0.0, im }
    }

    /// Create from polar coordinates r * e^(iθ).
    pub fn from_polar(r: f64, theta: f64) -> Self {
        Complex {
            re: r * theta.cos(),
            im: r * theta.sin(),
        }
    }

    /// Complex conjugate.
    pub fn conj(self) -> Self {
        Complex { re: self.re, im: -self.im }
    }

    /// Squared magnitude |z|².
    pub fn norm_sq(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    /// Magnitude |z|.
    pub fn norm(self) -> f64 {
        self.norm_sq().sqrt()
    }

    /// Argument (phase angle) in radians.
    pub fn arg(self) -> f64 {
        self.im.atan2(self.re)
    }

    /// Complex exponential e^z.
    pub fn exp(self) -> Self {
        let r = self.re.exp();
        Complex {
            re: r * self.im.cos(),
            im: r * self.im.sin(),
        }
    }

    /// Multiplicative inverse 1/z.
    pub fn inv(self) -> Self {
        let d = self.norm_sq();
        assert!(d > 1e-15, "Cannot invert zero complex number");
        Complex {
            re: self.re / d,
            im: -self.im / d,
        }
    }

    /// Complex division self / other.
    pub fn div(self, other: Self) -> Self {
        self * other.inv()
    }

    /// Approximate equality with tolerance.
    pub fn approx_eq(self, other: Self, tol: f64) -> bool {
        (self.re - other.re).abs() < tol && (self.im - other.im).abs() < tol
    }

    /// n-th root of unity e^(2πi/n).
    pub fn root_of_unity(n: usize) -> Self {
        let theta = 2.0 * PI / (n as f64);
        Complex::from_polar(1.0, theta)
    }

    /// k-th power of the n-th root of unity e^(2πik/n).
    pub fn root_of_unity_pow(n: usize, k: i64) -> Self {
        let theta = 2.0 * PI * (k as f64) / (n as f64);
        Complex::from_polar(1.0, theta)
    }

    /// Raise to integer power.
    pub fn powi(self, n: i32) -> Self {
        if n == 0 {
            return Complex::one();
        }
        if n < 0 {
            return self.inv().powi(-n);
        }
        let mut result = Complex::one();
        let mut base = self;
        let mut exp = n as u32;
        while exp > 0 {
            if exp & 1 == 1 {
                result = result * base;
            }
            base = base * base;
            exp >>= 1;
        }
        result
    }
}

impl fmt::Debug for Complex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.im >= 0.0 {
            write!(f, "{:.4}+{:.4}i", self.re, self.im)
        } else {
            write!(f, "{:.4}{:.4}i", self.re, self.im)
        }
    }
}

impl fmt::Display for Complex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.im.abs() < 1e-10 {
            write!(f, "{:.4}", self.re)
        } else if self.re.abs() < 1e-10 {
            write!(f, "{:.4}i", self.im)
        } else if self.im >= 0.0 {
            write!(f, "{:.4}+{:.4}i", self.re, self.im)
        } else {
            write!(f, "{:.4}{:.4}i", self.re, self.im)
        }
    }
}

impl PartialEq for Complex {
    fn eq(&self, other: &Self) -> bool {
        self.approx_eq(*other, 1e-10)
    }
}

impl std::ops::Add for Complex {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Complex { re: self.re + rhs.re, im: self.im + rhs.im }
    }
}

impl std::ops::Sub for Complex {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Complex { re: self.re - rhs.re, im: self.im - rhs.im }
    }
}

impl std::ops::Mul for Complex {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Complex {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl std::ops::Neg for Complex {
    type Output = Self;
    fn neg(self) -> Self {
        Complex { re: -self.re, im: -self.im }
    }
}

impl std::ops::Mul<f64> for Complex {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Complex { re: self.re * rhs, im: self.im * rhs }
    }
}

impl std::ops::AddAssign for Complex {
    fn add_assign(&mut self, rhs: Self) {
        self.re += rhs.re;
        self.im += rhs.im;
    }
}

impl std::ops::MulAssign for Complex {
    fn mul_assign(&mut self, rhs: Self) {
        let re = self.re * rhs.re - self.im * rhs.im;
        let im = self.re * rhs.im + self.im * rhs.re;
        self.re = re;
        self.im = im;
    }
}

// ═══════════════════════════════════════════════════════════════════════
// RepresentationField trait
// ═══════════════════════════════════════════════════════════════════════

/// Trait for scalar fields used in representations.
pub trait RepresentationField:
    Clone + Copy + PartialEq + fmt::Debug
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Neg<Output = Self>
    + std::ops::AddAssign
    + std::ops::MulAssign
{
    /// Additive identity.
    fn zero() -> Self;
    /// Multiplicative identity.
    fn one() -> Self;
    /// Multiplicative inverse.
    fn inv(self) -> Self;
    /// Conjugate (identity for reals, complex conjugate for Complex).
    fn conj(self) -> Self;
    /// Absolute value / norm.
    fn abs(self) -> f64;
    /// Convert from f64.
    fn from_f64(x: f64) -> Self;
}

impl RepresentationField for f64 {
    fn zero() -> Self { 0.0 }
    fn one() -> Self { 1.0 }
    fn inv(self) -> Self { 1.0 / self }
    fn conj(self) -> Self { self }
    fn abs(self) -> f64 { f64::abs(self) }
    fn from_f64(x: f64) -> Self { x }
}

impl RepresentationField for Complex {
    fn zero() -> Self { Complex::zero() }
    fn one() -> Self { Complex::one() }
    fn inv(self) -> Self { Complex::inv(self) }
    fn conj(self) -> Self { Complex::conj(self) }
    fn abs(self) -> f64 { self.norm() }
    fn from_f64(x: f64) -> Self { Complex::from_real(x) }
}

// ═══════════════════════════════════════════════════════════════════════
// Matrix<F> — dense matrix over a field
// ═══════════════════════════════════════════════════════════════════════

/// Dense matrix over a representation field.
#[derive(Clone)]
pub struct Matrix<F: RepresentationField> {
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
    /// Row-major storage.
    pub data: Vec<F>,
}

impl<F: RepresentationField> Matrix<F> {
    /// Create a zero matrix of given dimensions.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![F::zero(); rows * cols],
        }
    }

    /// Create an identity matrix.
    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.data[i * n + i] = F::one();
        }
        m
    }

    /// Create from a flat row-major vector.
    pub fn from_vec(rows: usize, cols: usize, data: Vec<F>) -> Self {
        assert_eq!(data.len(), rows * cols);
        Matrix { rows, cols, data }
    }

    /// Create from nested rows.
    pub fn from_rows(rows_data: &[Vec<F>]) -> Self {
        let rows = rows_data.len();
        assert!(rows > 0);
        let cols = rows_data[0].len();
        let mut data = Vec::with_capacity(rows * cols);
        for row in rows_data {
            assert_eq!(row.len(), cols);
            data.extend_from_slice(row);
        }
        Matrix { rows, cols, data }
    }

    /// Get element at (i, j).
    pub fn get(&self, i: usize, j: usize) -> F {
        self.data[i * self.cols + j]
    }

    /// Set element at (i, j).
    pub fn set(&mut self, i: usize, j: usize, val: F) {
        self.data[i * self.cols + j] = val;
    }

    /// Is this a square matrix?
    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    /// Matrix addition.
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let data: Vec<F> = self.data.iter().zip(&other.data)
            .map(|(&a, &b)| a + b)
            .collect();
        Matrix { rows: self.rows, cols: self.cols, data }
    }

    /// Matrix subtraction.
    pub fn sub(&self, other: &Self) -> Self {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let data: Vec<F> = self.data.iter().zip(&other.data)
            .map(|(&a, &b)| a - b)
            .collect();
        Matrix { rows: self.rows, cols: self.cols, data }
    }

    /// Matrix multiplication.
    pub fn mul(&self, other: &Self) -> Self {
        assert_eq!(self.cols, other.rows);
        let mut result = Self::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a = self.get(i, k);
                for j in 0..other.cols {
                    let val = result.get(i, j) + a * other.get(k, j);
                    result.set(i, j, val);
                }
            }
        }
        result
    }

    /// Scalar multiplication.
    pub fn scalar_mul(&self, s: F) -> Self {
        let data: Vec<F> = self.data.iter().map(|&x| x * s).collect();
        Matrix { rows: self.rows, cols: self.cols, data }
    }

    /// Transpose.
    pub fn transpose(&self) -> Self {
        let mut result = Self::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(j, i, self.get(i, j));
            }
        }
        result
    }

    /// Conjugate transpose (Hermitian adjoint).
    pub fn adjoint(&self) -> Self {
        let mut result = Self::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(j, i, self.get(i, j).conj());
            }
        }
        result
    }

    /// Trace of a square matrix.
    pub fn trace(&self) -> F {
        assert!(self.is_square());
        let mut t = F::zero();
        for i in 0..self.rows {
            t += self.get(i, i);
        }
        t
    }

    /// Direct sum of two matrices (block diagonal).
    pub fn direct_sum(&self, other: &Self) -> Self {
        let n = self.rows + other.rows;
        let m = self.cols + other.cols;
        let mut result = Self::zeros(n, m);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(i, j, self.get(i, j));
            }
        }
        for i in 0..other.rows {
            for j in 0..other.cols {
                result.set(self.rows + i, self.cols + j, other.get(i, j));
            }
        }
        result
    }

    /// Kronecker (tensor) product.
    pub fn kronecker(&self, other: &Self) -> Self {
        let rows = self.rows * other.rows;
        let cols = self.cols * other.cols;
        let mut result = Self::zeros(rows, cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                let s = self.get(i, j);
                for p in 0..other.rows {
                    for q in 0..other.cols {
                        result.set(
                            i * other.rows + p,
                            j * other.cols + q,
                            s * other.get(p, q),
                        );
                    }
                }
            }
        }
        result
    }

    /// Frobenius norm ||M||_F.
    pub fn frobenius_norm(&self) -> f64 {
        let mut s = 0.0;
        for &x in &self.data {
            s += x.abs() * x.abs();
        }
        s.sqrt()
    }

    /// Check if approximately equal to another matrix.
    pub fn approx_eq(&self, other: &Self, tol: f64) -> bool {
        if self.rows != other.rows || self.cols != other.cols {
            return false;
        }
        for i in 0..self.data.len() {
            if (self.data[i] - other.data[i]).abs() > tol {
                return false;
            }
        }
        true
    }

    /// Determinant of a square matrix (via LU-like expansion).
    pub fn determinant(&self) -> F {
        assert!(self.is_square());
        let n = self.rows;
        if n == 0 { return F::one(); }
        if n == 1 { return self.get(0, 0); }
        if n == 2 {
            return self.get(0, 0) * self.get(1, 1) - self.get(0, 1) * self.get(1, 0);
        }
        // Cofactor expansion along first row
        let mut det = F::zero();
        for j in 0..n {
            let minor = self.minor(0, j);
            let cofactor = minor.determinant();
            let sign = if j % 2 == 0 { F::one() } else { -F::one() };
            det += sign * self.get(0, j) * cofactor;
        }
        det
    }

    /// Minor matrix (delete row i, column j).
    pub fn minor(&self, row: usize, col: usize) -> Self {
        let n = self.rows - 1;
        let m = self.cols - 1;
        let mut result = Self::zeros(n, m);
        let mut ri = 0;
        for i in 0..self.rows {
            if i == row { continue; }
            let mut ci = 0;
            for j in 0..self.cols {
                if j == col { continue; }
                result.set(ri, ci, self.get(i, j));
                ci += 1;
            }
            ri += 1;
        }
        result
    }

    /// Inverse of a square matrix (via adjugate for small, Gauss-Jordan for larger).
    pub fn inverse(&self) -> Option<Self> {
        assert!(self.is_square());
        let n = self.rows;
        if n == 0 { return Some(Self::zeros(0, 0)); }

        // Gauss-Jordan elimination
        let mut aug = Self::zeros(n, 2 * n);
        for i in 0..n {
            for j in 0..n {
                aug.set(i, j, self.get(i, j));
            }
            aug.set(i, n + i, F::one());
        }

        for col in 0..n {
            // Find pivot
            let mut pivot_row = None;
            for row in col..n {
                if aug.get(row, col).abs() > 1e-12 {
                    pivot_row = Some(row);
                    break;
                }
            }
            let pivot_row = pivot_row?;

            // Swap rows
            if pivot_row != col {
                for j in 0..(2 * n) {
                    let tmp = aug.get(col, j);
                    aug.set(col, j, aug.get(pivot_row, j));
                    aug.set(pivot_row, j, tmp);
                }
            }

            // Scale pivot row
            let pivot = aug.get(col, col);
            let pivot_inv = pivot.inv();
            for j in 0..(2 * n) {
                let val = aug.get(col, j) * pivot_inv;
                aug.set(col, j, val);
            }

            // Eliminate column
            for row in 0..n {
                if row == col { continue; }
                let factor = aug.get(row, col);
                for j in 0..(2 * n) {
                    let val = aug.get(row, j) - factor * aug.get(col, j);
                    aug.set(row, j, val);
                }
            }
        }

        let mut result = Self::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                result.set(i, j, aug.get(i, n + j));
            }
        }
        Some(result)
    }

    /// Construct a permutation matrix from a Permutation.
    pub fn from_permutation(perm: &Permutation) -> Self {
        let n = perm.degree();
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.set(i, perm.apply(i as u32) as usize, F::one());
        }
        m
    }
}

impl<F: RepresentationField> fmt::Debug for Matrix<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Matrix({}x{}):", self.rows, self.cols)?;
        for i in 0..self.rows {
            write!(f, "  [")?;
            for j in 0..self.cols {
                if j > 0 { write!(f, ", ")?; }
                write!(f, "{:?}", self.get(i, j))?;
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}

impl<F: RepresentationField> PartialEq for Matrix<F> {
    fn eq(&self, other: &Self) -> bool {
        self.approx_eq(other, 1e-10)
    }
}

// ═══════════════════════════════════════════════════════════════════════
// ConjugacyClass — conjugacy class of a group element
// ═══════════════════════════════════════════════════════════════════════

/// A conjugacy class of elements in a group.
#[derive(Debug, Clone)]
pub struct ConjugacyClass {
    /// A representative element of the class.
    pub representative: Permutation,
    /// All elements in the conjugacy class.
    pub elements: Vec<Permutation>,
    /// The cycle type of elements in this class.
    pub cycle_type: Vec<usize>,
}

impl ConjugacyClass {
    /// Size of the conjugacy class.
    pub fn size(&self) -> usize {
        self.elements.len()
    }
}

/// Compute all conjugacy classes of a permutation group.
pub fn compute_conjugacy_classes(group: &PermutationGroup) -> Vec<ConjugacyClass> {
    let elements = group.enumerate_elements();
    let n = elements.len();
    let mut visited = vec![false; n];
    let mut classes = Vec::new();

    // Build index for fast lookup
    let elem_index: HashMap<Vec<u32>, usize> = elements.iter().enumerate()
        .map(|(i, p)| {
            let imgs: Vec<u32> = (0..p.degree() as u32).map(|x| p.apply(x)).collect();
            (imgs, i)
        })
        .collect();

    for i in 0..n {
        if visited[i] { continue; }

        let mut class_elems = Vec::new();
        for g in &elements {
            // Compute g * elements[i] * g^(-1)
            let conj = g.compose(&elements[i]).compose(&g.inverse());
            let imgs: Vec<u32> = (0..conj.degree() as u32).map(|x| conj.apply(x)).collect();
            if let Some(&idx) = elem_index.get(&imgs) {
                if !visited[idx] {
                    visited[idx] = true;
                    class_elems.push(elements[idx].clone());
                }
            }
        }

        let cycle_type = class_elems[0].cycle_type();
        classes.push(ConjugacyClass {
            representative: class_elems[0].clone(),
            cycle_type,
            elements: class_elems,
        });
    }

    classes
}

// ═══════════════════════════════════════════════════════════════════════
// GroupRepresentation — matrix representation of a group
// ═══════════════════════════════════════════════════════════════════════

/// A matrix representation of a finite group.
///
/// Maps each group element to an invertible matrix such that
/// ρ(g·h) = ρ(g) · ρ(h) for all group elements g, h.
#[derive(Debug, Clone)]
pub struct GroupRepresentation<F: RepresentationField> {
    /// Dimension of the representation.
    pub dimension: usize,
    /// Map from group element (as image vector) to matrix.
    images: HashMap<Vec<u32>, Matrix<F>>,
    /// Degree of the permutation group.
    pub degree: usize,
}

impl<F: RepresentationField> GroupRepresentation<F> {
    /// Create a new representation from a map of elements to matrices.
    pub fn new(degree: usize, dimension: usize, images: HashMap<Vec<u32>, Matrix<F>>) -> Self {
        GroupRepresentation { dimension, images, degree }
    }

    /// Get the matrix for a group element.
    pub fn matrix_for(&self, perm: &Permutation) -> Option<&Matrix<F>> {
        let key: Vec<u32> = (0..perm.degree() as u32).map(|x| perm.apply(x)).collect();
        self.images.get(&key)
    }

    /// Compute the character (trace) for a group element.
    pub fn character_of(&self, perm: &Permutation) -> F {
        self.matrix_for(perm)
            .map(|m| m.trace())
            .unwrap_or(F::zero())
    }

    /// Construct the natural (defining) permutation representation.
    /// Maps each permutation to its permutation matrix.
    pub fn natural_representation(group: &PermutationGroup) -> Self {
        let n = group.degree();
        let elements = group.enumerate_elements();
        let mut images = HashMap::new();
        for elem in &elements {
            let key: Vec<u32> = (0..n as u32).map(|x| elem.apply(x)).collect();
            images.insert(key, Matrix::<F>::from_permutation(elem));
        }
        GroupRepresentation { dimension: n, images, degree: n }
    }

    /// Construct the trivial (1-dimensional) representation.
    /// Maps every element to the 1×1 identity matrix.
    pub fn trivial_representation(group: &PermutationGroup) -> Self {
        let elements = group.enumerate_elements();
        let mut images = HashMap::new();
        for elem in &elements {
            let key: Vec<u32> = (0..elem.degree() as u32).map(|x| elem.apply(x)).collect();
            images.insert(key, Matrix::<F>::identity(1));
        }
        GroupRepresentation { dimension: 1, images, degree: group.degree() }
    }

    /// Construct the sign (alternating) representation.
    /// Maps each permutation to its sign (+1 or -1) as a 1×1 matrix.
    pub fn sign_representation(group: &PermutationGroup) -> Self {
        let elements = group.enumerate_elements();
        let mut images = HashMap::new();
        for elem in &elements {
            let key: Vec<u32> = (0..elem.degree() as u32).map(|x| elem.apply(x)).collect();
            let sign = if elem.sign() == 1 { F::one() } else { -F::one() };
            images.insert(key, Matrix::from_vec(1, 1, vec![sign]));
        }
        GroupRepresentation { dimension: 1, images, degree: group.degree() }
    }

    /// Regular representation (group acting on itself by left multiplication).
    pub fn regular_representation(group: &PermutationGroup) -> Self {
        let elements = group.enumerate_elements();
        let n = elements.len();
        let elem_index: HashMap<Vec<u32>, usize> = elements.iter().enumerate()
            .map(|(i, p)| {
                let key: Vec<u32> = (0..p.degree() as u32).map(|x| p.apply(x)).collect();
                (key, i)
            })
            .collect();

        let mut images = HashMap::new();
        for g in &elements {
            let mut mat = Matrix::<F>::zeros(n, n);
            for (j, h) in elements.iter().enumerate() {
                let gh = g.compose(h);
                let key: Vec<u32> = (0..gh.degree() as u32).map(|x| gh.apply(x)).collect();
                if let Some(&i) = elem_index.get(&key) {
                    mat.set(i, j, F::one());
                }
            }
            let gkey: Vec<u32> = (0..g.degree() as u32).map(|x| g.apply(x)).collect();
            images.insert(gkey, mat);
        }

        GroupRepresentation { dimension: n, images, degree: group.degree() }
    }

    /// Direct sum of two representations.
    pub fn direct_sum(&self, other: &Self) -> Self {
        assert_eq!(self.degree, other.degree);
        let dim = self.dimension + other.dimension;
        let mut images = HashMap::new();
        for (key, m1) in &self.images {
            if let Some(m2) = other.images.get(key) {
                images.insert(key.clone(), m1.direct_sum(m2));
            }
        }
        GroupRepresentation { dimension: dim, images, degree: self.degree }
    }

    /// Tensor product of two representations.
    pub fn tensor_product(&self, other: &Self) -> Self {
        assert_eq!(self.degree, other.degree);
        let dim = self.dimension * other.dimension;
        let mut images = HashMap::new();
        for (key, m1) in &self.images {
            if let Some(m2) = other.images.get(key) {
                images.insert(key.clone(), m1.kronecker(m2));
            }
        }
        GroupRepresentation { dimension: dim, images, degree: self.degree }
    }

    /// Dual (contragredient) representation.
    pub fn dual(&self) -> Self {
        let mut images = HashMap::new();
        for (key, m) in &self.images {
            if let Some(inv) = m.inverse() {
                images.insert(key.clone(), inv.adjoint());
            }
        }
        GroupRepresentation {
            dimension: self.dimension,
            images,
            degree: self.degree,
        }
    }

    /// Check if this is a valid representation (homomorphism property).
    pub fn is_valid(&self, group: &PermutationGroup) -> bool {
        let elements = group.enumerate_elements();
        for g in &elements {
            for h in &elements {
                let gh = g.compose(h);
                let mg = match self.matrix_for(g) { Some(m) => m, None => return false };
                let mh = match self.matrix_for(h) { Some(m) => m, None => return false };
                let mgh = match self.matrix_for(&gh) { Some(m) => m, None => return false };
                let product = mg.mul(mh);
                if !product.approx_eq(mgh, 1e-8) {
                    return false;
                }
            }
        }
        true
    }

    /// Number of elements in the representation.
    pub fn num_elements(&self) -> usize {
        self.images.len()
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Character — trace function on a representation
// ═══════════════════════════════════════════════════════════════════════

/// Character of a representation: maps conjugacy classes to field values.
#[derive(Debug, Clone)]
pub struct Character<F: RepresentationField> {
    /// Dimension of the representation.
    pub dimension: usize,
    /// Character values indexed by conjugacy class representative (as image vec).
    pub values: BTreeMap<Vec<u32>, F>,
}

impl<F: RepresentationField> Character<F> {
    /// Create a character from a representation and conjugacy classes.
    pub fn from_representation(
        rep: &GroupRepresentation<F>,
        classes: &[ConjugacyClass],
    ) -> Self {
        let mut values = BTreeMap::new();
        for class in classes {
            let val = rep.character_of(&class.representative);
            let key: Vec<u32> = (0..class.representative.degree() as u32)
                .map(|x| class.representative.apply(x))
                .collect();
            values.insert(key, val);
        }
        Character {
            dimension: rep.dimension,
            values,
        }
    }

    /// Inner product of two characters: ⟨χ, ψ⟩ = (1/|G|) Σ |C_i| χ(g_i) ψ(g_i)*
    pub fn inner_product(
        &self,
        other: &Character<F>,
        classes: &[ConjugacyClass],
        group_order: u64,
    ) -> F {
        let mut sum = F::zero();
        for class in classes {
            let key: Vec<u32> = (0..class.representative.degree() as u32)
                .map(|x| class.representative.apply(x))
                .collect();
            if let (Some(&chi), Some(&psi)) = (self.values.get(&key), other.values.get(&key)) {
                let class_size = F::from_f64(class.size() as f64);
                sum += class_size * chi * psi.conj();
            }
        }
        sum * F::from_f64(1.0 / group_order as f64)
    }

    /// Check if this character is irreducible: ⟨χ, χ⟩ = 1.
    pub fn is_irreducible(&self, classes: &[ConjugacyClass], group_order: u64) -> bool {
        let ip = self.inner_product(self, classes, group_order);
        ip.abs() - 1.0 < 0.01
    }

    /// Multiplicity of an irreducible character in this character.
    pub fn multiplicity(
        &self,
        irreducible: &Character<F>,
        classes: &[ConjugacyClass],
        group_order: u64,
    ) -> usize {
        let ip = self.inner_product(irreducible, classes, group_order);
        ip.abs().round() as usize
    }

    /// Add two characters (direct sum).
    pub fn add(&self, other: &Character<F>) -> Self {
        let mut values = BTreeMap::new();
        for (key, &val) in &self.values {
            let other_val = other.values.get(key).copied().unwrap_or(F::zero());
            values.insert(key.clone(), val + other_val);
        }
        Character {
            dimension: self.dimension + other.dimension,
            values,
        }
    }

    /// Multiply two characters (tensor product).
    pub fn mul(&self, other: &Character<F>) -> Self {
        let mut values = BTreeMap::new();
        for (key, &val) in &self.values {
            let other_val = other.values.get(key).copied().unwrap_or(F::zero());
            values.insert(key.clone(), val * other_val);
        }
        Character {
            dimension: self.dimension * other.dimension,
            values,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// CharacterTable — complete character table of a finite group
// ═══════════════════════════════════════════════════════════════════════

/// Complete character table of a finite group.
#[derive(Debug, Clone)]
pub struct CharacterTable {
    /// Conjugacy classes.
    pub classes: Vec<ConjugacyClass>,
    /// Irreducible characters (as Complex-valued).
    pub characters: Vec<Character<Complex>>,
    /// Order of the group.
    pub group_order: u64,
}

impl CharacterTable {
    /// Compute the character table for a small group.
    ///
    /// Uses the fact that for an abelian group all irreps are 1-dimensional,
    /// and for general groups we decompose the regular representation.
    pub fn compute(group: &PermutationGroup) -> Self {
        let classes = compute_conjugacy_classes(group);
        let group_order = group.order();
        let num_classes = classes.len();

        // For small groups, compute characters from natural + sign representations
        // and use orthogonality to find all irreducible characters.
        let mut characters: Vec<Character<Complex>> = Vec::new();

        // Trivial character
        let trivial = GroupRepresentation::<Complex>::trivial_representation(group);
        let chi_trivial = Character::from_representation(&trivial, &classes);
        characters.push(chi_trivial);

        // Sign character (for symmetric groups)
        let sign = GroupRepresentation::<Complex>::sign_representation(group);
        let chi_sign = Character::from_representation(&sign, &classes);
        if characters.iter().all(|c| c.values != chi_sign.values) {
            characters.push(chi_sign);
        }

        // Standard representation (natural minus trivial)
        if group.degree() > 1 {
            let nat = GroupRepresentation::<Complex>::natural_representation(group);
            let chi_nat = Character::from_representation(&nat, &classes);

            // Standard = natural - trivial
            let mut std_values = BTreeMap::new();
            for (key, &val) in &chi_nat.values {
                let triv_val = characters[0].values.get(key).copied().unwrap_or(Complex::zero());
                std_values.insert(key.clone(), val - triv_val);
            }
            let chi_std = Character {
                dimension: group.degree() - 1,
                values: std_values,
            };
            if chi_std.is_irreducible(&classes, group_order) {
                characters.push(chi_std.clone());
            }

            // Sign ⊗ standard
            if characters.len() > 1 {
                let chi_sign_std = characters[1].mul(&chi_std);
                if chi_sign_std.is_irreducible(&classes, group_order)
                    && characters.iter().all(|c| {
                        let diff: f64 = c.values.iter().zip(chi_sign_std.values.iter())
                            .map(|((_, &a), (_, &b))| (a - b).norm())
                            .sum();
                        diff > 0.1
                    })
                {
                    characters.push(chi_sign_std);
                }
            }
        }

        // For abelian groups, all irreps are 1-dimensional
        if group_order == num_classes as u64 {
            characters.clear();
            // All elements have order dividing |G|
            let elements = group.enumerate_elements();
            let elem_index: HashMap<Vec<u32>, usize> = elements.iter().enumerate()
                .map(|(i, p)| {
                    let key: Vec<u32> = (0..p.degree() as u32).map(|x| p.apply(x)).collect();
                    (key, i)
                })
                .collect();

            // For cyclic group Z_n, character χ_k(g^j) = ω^(jk), ω = e^(2πi/n)
            // For general abelian, decompose into cyclic factors
            // Simple approach: enumerate all homomorphisms G -> C*
            let n = group_order;
            for k in 0..n {
                let mut values = BTreeMap::new();
                for class in &classes {
                    let key: Vec<u32> = (0..class.representative.degree() as u32)
                        .map(|x| class.representative.apply(x))
                        .collect();
                    // For cyclic groups, use powers of root of unity
                    let order = class.representative.order();
                    let power = k % order;
                    let val = Complex::root_of_unity_pow(order as usize, power as i64);
                    values.insert(key, val);
                }
                let chi = Character { dimension: 1, values };
                // Check it's actually a character (homomorphism property)
                characters.push(chi);
            }
            // Keep only distinct irreducible ones
            let mut unique = Vec::new();
            for chi in &characters {
                if chi.is_irreducible(&classes, group_order) {
                    let is_dup = unique.iter().any(|c: &Character<Complex>| {
                        let diff: f64 = c.values.iter().zip(chi.values.iter())
                            .map(|((_, &a), (_, &b))| (a - b).norm())
                            .sum();
                        diff < 0.1
                    });
                    if !is_dup {
                        unique.push(chi.clone());
                    }
                }
            }
            characters = unique;
        }

        // Pad with zero characters if we haven't found enough
        while characters.len() < num_classes {
            let mut values = BTreeMap::new();
            for class in &classes {
                let key: Vec<u32> = (0..class.representative.degree() as u32)
                    .map(|x| class.representative.apply(x))
                    .collect();
                values.insert(key, Complex::zero());
            }
            characters.push(Character { dimension: 0, values });
        }

        CharacterTable {
            classes,
            characters,
            group_order,
        }
    }

    /// Get the number of conjugacy classes (= number of irreducible representations).
    pub fn num_classes(&self) -> usize {
        self.classes.len()
    }

    /// Get character value χ_i(C_j).
    pub fn value(&self, char_idx: usize, class_idx: usize) -> Complex {
        let key: Vec<u32> = (0..self.classes[class_idx].representative.degree() as u32)
            .map(|x| self.classes[class_idx].representative.apply(x))
            .collect();
        self.characters[char_idx].values.get(&key).copied().unwrap_or(Complex::zero())
    }

    /// Display the character table as a formatted string.
    pub fn display(&self) -> String {
        let mut s = String::new();
        s.push_str("Character Table:\n");
        s.push_str(&format!("Group order: {}\n", self.group_order));
        s.push_str(&format!("Classes: {}\n\n", self.num_classes()));

        // Header
        s.push_str("       ");
        for (i, class) in self.classes.iter().enumerate() {
            s.push_str(&format!(" C{:2}({:2})", i, class.size()));
        }
        s.push('\n');

        // Character rows
        for (i, chi) in self.characters.iter().enumerate() {
            s.push_str(&format!("χ_{:<3}  ", i));
            for class in &self.classes {
                let key: Vec<u32> = (0..class.representative.degree() as u32)
                    .map(|x| class.representative.apply(x))
                    .collect();
                let val = chi.values.get(&key).copied().unwrap_or(Complex::zero());
                s.push_str(&format!(" {:>7}", format!("{}", val)));
            }
            s.push('\n');
        }

        s
    }
}

// ═══════════════════════════════════════════════════════════════════════
// IrreducibleDecomposition
// ═══════════════════════════════════════════════════════════════════════

/// Decomposition of a representation into irreducible constituents.
#[derive(Debug, Clone)]
pub struct IrreducibleDecomposition {
    /// Multiplicities of each irreducible representation.
    pub multiplicities: Vec<usize>,
    /// Total dimension.
    pub total_dimension: usize,
}

impl IrreducibleDecomposition {
    /// Decompose a character into irreducible characters using inner products.
    pub fn decompose(
        character: &Character<Complex>,
        table: &CharacterTable,
    ) -> Self {
        let mut multiplicities = Vec::new();
        let mut total_dim = 0;
        for irrep in &table.characters {
            let mult = character.multiplicity(irrep, &table.classes, table.group_order);
            multiplicities.push(mult);
            total_dim += mult * irrep.dimension;
        }
        IrreducibleDecomposition {
            multiplicities,
            total_dimension: total_dim,
        }
    }

    /// Check if the representation is irreducible.
    pub fn is_irreducible(&self) -> bool {
        self.multiplicities.iter().filter(|&&m| m > 0).count() == 1
            && self.multiplicities.iter().any(|&m| m == 1)
    }

    /// Number of irreducible constituents (with multiplicity).
    pub fn num_constituents(&self) -> usize {
        self.multiplicities.iter().sum()
    }

    /// Display the decomposition.
    pub fn display(&self) -> String {
        let mut parts = Vec::new();
        for (i, &mult) in self.multiplicities.iter().enumerate() {
            if mult > 0 {
                if mult == 1 {
                    parts.push(format!("χ_{}", i));
                } else {
                    parts.push(format!("{}·χ_{}", mult, i));
                }
            }
        }
        if parts.is_empty() {
            "0".to_string()
        } else {
            parts.join(" ⊕ ")
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// BurnsideCounter — Burnside's lemma for counting orbits
// ═══════════════════════════════════════════════════════════════════════

/// Burnside's lemma: counts distinct objects under group action.
///
/// |X/G| = (1/|G|) Σ_{g∈G} |Fix(g)|
#[derive(Debug, Clone)]
pub struct BurnsideCounter {
    /// Group acting on the set.
    group_order: u64,
    /// Sum of fixed-point counts.
    fixed_point_sum: usize,
}

impl BurnsideCounter {
    /// Create a new Burnside counter for the given group.
    pub fn new(group: &PermutationGroup) -> Self {
        BurnsideCounter {
            group_order: group.order(),
            fixed_point_sum: 0,
        }
    }

    /// Count orbits using Burnside's lemma on a set of objects.
    ///
    /// `fixed_points` is a function that returns the number of objects
    /// fixed by a given group element.
    pub fn count_orbits<Fp>(group: &PermutationGroup, fixed_points: Fp) -> usize
    where
        Fp: Fn(&Permutation) -> usize,
    {
        let elements = group.enumerate_elements();
        let sum: usize = elements.iter().map(|g| fixed_points(g)).sum();
        sum / group.order() as usize
    }

    /// Count orbits on a set {0, 1, ..., n-1} under the natural action.
    pub fn count_point_orbits(group: &PermutationGroup) -> usize {
        Self::count_orbits(group, |g| {
            (0..g.degree() as u32).filter(|&x| g.apply(x) == x).count()
        })
    }

    /// Count orbits of k-element subsets of {0, ..., n-1}.
    pub fn count_subset_orbits(group: &PermutationGroup, k: usize) -> usize {
        let n = group.degree();
        if k > n { return 0; }

        // Generate all k-subsets
        let subsets = Self::gen_subsets(n, k);

        Self::count_orbits(group, |g| {
            subsets.iter().filter(|subset| {
                let mapped: HashSet<u32> = subset.iter().map(|&x| g.apply(x)).collect();
                let orig: HashSet<u32> = subset.iter().cloned().collect();
                mapped == orig
            }).count()
        })
    }

    fn gen_subsets(n: usize, k: usize) -> Vec<Vec<u32>> {
        let mut result = Vec::new();
        let mut current = Vec::new();
        Self::gen_subsets_rec(n as u32, k, 0, &mut current, &mut result);
        result
    }

    fn gen_subsets_rec(n: u32, k: usize, start: u32, current: &mut Vec<u32>, result: &mut Vec<Vec<u32>>) {
        if current.len() == k {
            result.push(current.clone());
            return;
        }
        for i in start..n {
            current.push(i);
            Self::gen_subsets_rec(n, k, i + 1, current, result);
            current.pop();
        }
    }

    /// Count colorings of n positions with c colors under the group action.
    /// Uses Burnside's lemma: each g fixes c^(number of cycles of g) colorings.
    pub fn count_colorings(group: &PermutationGroup, num_colors: usize) -> usize {
        Self::count_orbits(group, |g| {
            let num_cycles = g.cycle_type().len();
            num_colors.pow(num_cycles as u32)
        })
    }

    /// Accumulate a fixed-point count.
    pub fn add_fixed_points(&mut self, count: usize) {
        self.fixed_point_sum += count;
    }

    /// Get the current orbit count.
    pub fn orbit_count(&self) -> usize {
        self.fixed_point_sum / self.group_order as usize
    }
}

// ═══════════════════════════════════════════════════════════════════════
// SchurOrthogonality — verify Schur orthogonality relations
// ═══════════════════════════════════════════════════════════════════════

/// Schur orthogonality relations verification.
///
/// First orthogonality: Σ_g χ_i(g) χ_j(g)* = |G| δ_{ij}
/// Second orthogonality: Σ_i χ_i(g) χ_i(h)* = |G|/|C_g| δ_{C_g, C_h}
#[derive(Debug)]
pub struct SchurOrthogonality;

impl SchurOrthogonality {
    /// Verify the first (row) orthogonality relation.
    pub fn verify_first_orthogonality(table: &CharacterTable) -> bool {
        let tol = 0.01;
        for i in 0..table.characters.len() {
            for j in 0..table.characters.len() {
                let ip = table.characters[i].inner_product(
                    &table.characters[j],
                    &table.classes,
                    table.group_order,
                );
                let expected = if i == j { 1.0 } else { 0.0 };
                if (ip.abs() - expected).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    /// Verify the second (column) orthogonality relation.
    pub fn verify_second_orthogonality(table: &CharacterTable) -> bool {
        let tol = 0.1;
        let n = table.num_classes();
        for a in 0..n {
            for b in 0..n {
                let mut sum = Complex::zero();
                for i in 0..table.characters.len() {
                    let chi_a = table.value(i, a);
                    let chi_b = table.value(i, b);
                    sum += chi_a * chi_b.conj();
                }
                let expected = if a == b {
                    table.group_order as f64 / table.classes[a].size() as f64
                } else {
                    0.0
                };
                if (sum.norm() - expected).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    /// Check that sum of squares of dimensions equals group order.
    pub fn verify_dimension_formula(table: &CharacterTable) -> bool {
        let sum: usize = table.characters.iter()
            .map(|chi| chi.dimension * chi.dimension)
            .sum();
        sum == table.group_order as usize
    }
}

// ═══════════════════════════════════════════════════════════════════════
// FourierTransform — Fourier transform on finite groups
// ═══════════════════════════════════════════════════════════════════════

/// Fourier transform on finite groups.
///
/// For a function f: G → ℂ, the Fourier transform is:
///   f̂(ρ) = Σ_{g∈G} f(g) ρ(g)
/// where ρ ranges over irreducible representations.
#[derive(Debug, Clone)]
pub struct FourierTransform {
    /// Group order.
    group_order: u64,
}

impl FourierTransform {
    /// Create a new Fourier transform engine.
    pub fn new(group_order: u64) -> Self {
        FourierTransform { group_order }
    }

    /// Forward Fourier transform of a function on the group.
    ///
    /// Given f(g) for each group element g, compute f̂(ρ) = Σ f(g) ρ(g)
    /// for each irreducible representation ρ.
    pub fn forward(
        &self,
        function: &HashMap<Vec<u32>, Complex>,
        representations: &[GroupRepresentation<Complex>],
    ) -> Vec<Matrix<Complex>> {
        let mut result = Vec::new();
        for rep in representations {
            let dim = rep.dimension;
            let mut fhat = Matrix::<Complex>::zeros(dim, dim);
            for (key, &fval) in function {
                if let Some(rho) = rep.images.get(key) {
                    let scaled = rho.scalar_mul(fval);
                    fhat = fhat.add(&scaled);
                }
            }
            result.push(fhat);
        }
        result
    }

    /// Inverse Fourier transform.
    ///
    /// Given f̂(ρ_i) for each irreducible representation, recover f(g):
    ///   f(g) = (1/|G|) Σ_i d_i Tr(f̂(ρ_i) ρ_i(g⁻¹))
    pub fn inverse(
        &self,
        transforms: &[Matrix<Complex>],
        representations: &[GroupRepresentation<Complex>],
        elements: &[Permutation],
    ) -> HashMap<Vec<u32>, Complex> {
        let mut result = HashMap::new();
        let inv_order = Complex::from_real(1.0 / self.group_order as f64);

        for g in elements {
            let g_inv = g.inverse();
            let mut sum = Complex::zero();

            for (i, rep) in representations.iter().enumerate() {
                let dim = rep.dimension;
                let fhat = &transforms[i];
                if let Some(rho_ginv) = rep.matrix_for(&g_inv) {
                    let product = fhat.mul(rho_ginv);
                    let trace = product.trace();
                    sum += Complex::from_real(dim as f64) * trace;
                }
            }

            sum *= inv_order;
            let key: Vec<u32> = (0..g.degree() as u32).map(|x| g.apply(x)).collect();
            result.insert(key, sum);
        }

        result
    }

    /// Convolution of two functions on the group.
    ///
    /// (f * g)(x) = Σ_{y∈G} f(y) g(y⁻¹x)
    pub fn convolution(
        &self,
        f: &HashMap<Vec<u32>, Complex>,
        g: &HashMap<Vec<u32>, Complex>,
        elements: &[Permutation],
    ) -> HashMap<Vec<u32>, Complex> {
        let mut result = HashMap::new();

        for x in elements {
            let mut sum = Complex::zero();
            for y in elements {
                let y_inv = y.inverse();
                let y_inv_x = y_inv.compose(x);
                let key_y: Vec<u32> = (0..y.degree() as u32).map(|i| y.apply(i)).collect();
                let key_yx: Vec<u32> = (0..y_inv_x.degree() as u32)
                    .map(|i| y_inv_x.apply(i)).collect();
                let fval = f.get(&key_y).copied().unwrap_or(Complex::zero());
                let gval = g.get(&key_yx).copied().unwrap_or(Complex::zero());
                sum += fval * gval;
            }
            let key_x: Vec<u32> = (0..x.degree() as u32).map(|i| x.apply(i)).collect();
            result.insert(key_x, sum);
        }

        result
    }

    /// Plancherel formula: ||f||² = (1/|G|) Σ_i d_i ||f̂(ρ_i)||²_F
    pub fn plancherel_norm(
        &self,
        transforms: &[Matrix<Complex>],
        dimensions: &[usize],
    ) -> f64 {
        let mut sum = 0.0;
        for (i, fhat) in transforms.iter().enumerate() {
            sum += (dimensions[i] as f64) * fhat.frobenius_norm().powi(2);
        }
        sum / self.group_order as f64
    }
}

// ═══════════════════════════════════════════════════════════════════════
// InducedRepresentation
// ═══════════════════════════════════════════════════════════════════════

/// Construct an induced representation from a subgroup representation.
///
/// If H ≤ G and ρ: H → GL(V), then Ind_H^G(ρ) = ⊕_{g∈G/H} gV.
#[derive(Debug)]
pub struct InducedRepresentation;

impl InducedRepresentation {
    /// Compute the character of an induced representation.
    ///
    /// χ_{Ind}(g) = (1/|H|) Σ_{x∈G} χ_H(x⁻¹gx) where χ_H is extended by 0.
    pub fn induced_character(
        group: &PermutationGroup,
        subgroup_elements: &[Permutation],
        subgroup_character: &Character<Complex>,
        classes: &[ConjugacyClass],
    ) -> Character<Complex> {
        let subgroup_set: HashSet<Vec<u32>> = subgroup_elements.iter()
            .map(|h| {
                let key: Vec<u32> = (0..h.degree() as u32).map(|x| h.apply(x)).collect();
                key
            })
            .collect();

        let h_order = subgroup_elements.len();
        let g_elements = group.enumerate_elements();

        let mut values = BTreeMap::new();
        for class in classes {
            let g = &class.representative;
            let mut sum = Complex::zero();
            for x in &g_elements {
                let x_inv = x.inverse();
                let conj = x_inv.compose(g).compose(x);
                let conj_key: Vec<u32> = (0..conj.degree() as u32)
                    .map(|i| conj.apply(i)).collect();
                if subgroup_set.contains(&conj_key) {
                    if let Some(&val) = subgroup_character.values.get(&conj_key) {
                        sum += val;
                    }
                }
            }
            sum = sum * Complex::from_real(1.0 / h_order as f64);
            let key: Vec<u32> = (0..g.degree() as u32).map(|i| g.apply(i)).collect();
            values.insert(key, sum);
        }

        let dim = (group.order() as usize / h_order) * subgroup_character.dimension;
        Character { dimension: dim, values }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// RestrictedRepresentation
// ═══════════════════════════════════════════════════════════════════════

/// Restrict a representation to a subgroup.
#[derive(Debug)]
pub struct RestrictedRepresentation;

impl RestrictedRepresentation {
    /// Restrict a representation to a subgroup.
    pub fn restrict(
        rep: &GroupRepresentation<Complex>,
        subgroup_elements: &[Permutation],
    ) -> GroupRepresentation<Complex> {
        let mut images = HashMap::new();
        for h in subgroup_elements {
            let key: Vec<u32> = (0..h.degree() as u32).map(|x| h.apply(x)).collect();
            if let Some(mat) = rep.images.get(&key) {
                images.insert(key, mat.clone());
            }
        }
        GroupRepresentation {
            dimension: rep.dimension,
            images,
            degree: rep.degree,
        }
    }

    /// Restrict a character to a subgroup.
    pub fn restrict_character(
        character: &Character<Complex>,
        subgroup_classes: &[ConjugacyClass],
    ) -> Character<Complex> {
        let mut values = BTreeMap::new();
        for class in subgroup_classes {
            let key: Vec<u32> = (0..class.representative.degree() as u32)
                .map(|x| class.representative.apply(x))
                .collect();
            if let Some(&val) = character.values.get(&key) {
                values.insert(key, val);
            }
        }
        Character {
            dimension: character.dimension,
            values,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// TensorProduct
// ═══════════════════════════════════════════════════════════════════════

/// Tensor product of representations.
#[derive(Debug)]
pub struct TensorProduct;

impl TensorProduct {
    /// Compute the tensor product of two representations.
    pub fn compute(
        rep1: &GroupRepresentation<Complex>,
        rep2: &GroupRepresentation<Complex>,
    ) -> GroupRepresentation<Complex> {
        rep1.tensor_product(rep2)
    }

    /// Compute the character of a tensor product.
    pub fn character(
        chi1: &Character<Complex>,
        chi2: &Character<Complex>,
    ) -> Character<Complex> {
        chi1.mul(chi2)
    }

    /// Decompose a tensor product into irreducibles.
    pub fn decompose(
        chi1: &Character<Complex>,
        chi2: &Character<Complex>,
        table: &CharacterTable,
    ) -> IrreducibleDecomposition {
        let product = Self::character(chi1, chi2);
        IrreducibleDecomposition::decompose(&product, table)
    }

    /// Symmetric square of a representation character.
    /// Sym²(χ)(g) = (χ(g)² + χ(g²)) / 2
    pub fn symmetric_square(
        chi: &Character<Complex>,
        group: &PermutationGroup,
        classes: &[ConjugacyClass],
    ) -> Character<Complex> {
        let elements = group.enumerate_elements();
        let mut values = BTreeMap::new();

        for class in classes {
            let g = &class.representative;
            let g2 = g.compose(g);

            let g_key: Vec<u32> = (0..g.degree() as u32).map(|x| g.apply(x)).collect();
            let g2_key: Vec<u32> = (0..g2.degree() as u32).map(|x| g2.apply(x)).collect();

            let chi_g = chi.values.get(&g_key).copied().unwrap_or(Complex::zero());
            let chi_g2 = chi.values.get(&g2_key).copied().unwrap_or(Complex::zero());

            let val = (chi_g * chi_g + chi_g2) * Complex::from_real(0.5);
            values.insert(g_key, val);
        }

        let dim = chi.dimension * (chi.dimension + 1) / 2;
        Character { dimension: dim, values }
    }

    /// Exterior (alternating) square of a representation character.
    /// Λ²(χ)(g) = (χ(g)² - χ(g²)) / 2
    pub fn exterior_square(
        chi: &Character<Complex>,
        group: &PermutationGroup,
        classes: &[ConjugacyClass],
    ) -> Character<Complex> {
        let elements = group.enumerate_elements();
        let mut values = BTreeMap::new();

        for class in classes {
            let g = &class.representative;
            let g2 = g.compose(g);

            let g_key: Vec<u32> = (0..g.degree() as u32).map(|x| g.apply(x)).collect();
            let g2_key: Vec<u32> = (0..g2.degree() as u32).map(|x| g2.apply(x)).collect();

            let chi_g = chi.values.get(&g_key).copied().unwrap_or(Complex::zero());
            let chi_g2 = chi.values.get(&g2_key).copied().unwrap_or(Complex::zero());

            let val = (chi_g * chi_g - chi_g2) * Complex::from_real(0.5);
            values.insert(g_key, val);
        }

        let dim = chi.dimension * (chi.dimension - 1) / 2;
        Character { dimension: dim, values }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// RepresentationRing — formal ring of virtual representations
// ═══════════════════════════════════════════════════════════════════════

/// The representation ring R(G) of a finite group.
///
/// Elements are formal ℤ-linear combinations of irreducible representations.
/// Addition = direct sum, multiplication = tensor product.
#[derive(Debug, Clone)]
pub struct RepresentationRing {
    /// Character table of the group.
    table: CharacterTable,
}

/// An element of the representation ring: a vector of multiplicities.
#[derive(Debug, Clone, PartialEq)]
pub struct RingElement {
    /// Multiplicities (can be negative for virtual representations).
    pub coefficients: Vec<i64>,
}

impl RepresentationRing {
    /// Create the representation ring for a group.
    pub fn new(table: CharacterTable) -> Self {
        RepresentationRing { table }
    }

    /// The zero element.
    pub fn zero(&self) -> RingElement {
        RingElement {
            coefficients: vec![0; self.table.num_classes()],
        }
    }

    /// The i-th irreducible representation.
    pub fn irreducible(&self, i: usize) -> RingElement {
        let mut coeffs = vec![0i64; self.table.num_classes()];
        coeffs[i] = 1;
        RingElement { coefficients: coeffs }
    }

    /// The regular representation.
    pub fn regular(&self) -> RingElement {
        let coeffs: Vec<i64> = self.table.characters.iter()
            .map(|chi| chi.dimension as i64)
            .collect();
        RingElement { coefficients: coeffs }
    }

    /// Add two ring elements (direct sum).
    pub fn add(&self, a: &RingElement, b: &RingElement) -> RingElement {
        let coeffs: Vec<i64> = a.coefficients.iter().zip(&b.coefficients)
            .map(|(&x, &y)| x + y)
            .collect();
        RingElement { coefficients: coeffs }
    }

    /// Subtract ring elements.
    pub fn sub(&self, a: &RingElement, b: &RingElement) -> RingElement {
        let coeffs: Vec<i64> = a.coefficients.iter().zip(&b.coefficients)
            .map(|(&x, &y)| x - y)
            .collect();
        RingElement { coefficients: coeffs }
    }

    /// Multiply two ring elements (tensor product).
    pub fn mul(&self, a: &RingElement, b: &RingElement) -> RingElement {
        // Reconstruct characters, multiply, decompose
        let chi_a = self.to_character(a);
        let chi_b = self.to_character(b);
        let product = chi_a.mul(&chi_b);
        let decomp = IrreducibleDecomposition::decompose(&product, &self.table);
        RingElement {
            coefficients: decomp.multiplicities.iter().map(|&m| m as i64).collect(),
        }
    }

    /// Dimension of a ring element.
    pub fn dimension(&self, elem: &RingElement) -> i64 {
        elem.coefficients.iter().zip(&self.table.characters)
            .map(|(&c, chi)| c * chi.dimension as i64)
            .sum()
    }

    /// Convert a ring element to a character.
    fn to_character(&self, elem: &RingElement) -> Character<Complex> {
        let mut values = BTreeMap::new();
        for class in &self.table.classes {
            let key: Vec<u32> = (0..class.representative.degree() as u32)
                .map(|x| class.representative.apply(x))
                .collect();
            let mut val = Complex::zero();
            for (i, &coeff) in elem.coefficients.iter().enumerate() {
                if let Some(&chi_val) = self.table.characters[i].values.get(&key) {
                    val += chi_val * Complex::from_real(coeff as f64);
                }
            }
            values.insert(key, val);
        }
        let dim: usize = elem.coefficients.iter().zip(&self.table.characters)
            .map(|(&c, chi)| (c as usize) * chi.dimension)
            .sum();
        Character { dimension: dim, values }
    }

    /// Display a ring element.
    pub fn display(&self, elem: &RingElement) -> String {
        let mut parts = Vec::new();
        for (i, &coeff) in elem.coefficients.iter().enumerate() {
            if coeff > 0 {
                if coeff == 1 {
                    parts.push(format!("χ_{}", i));
                } else {
                    parts.push(format!("{}·χ_{}", coeff, i));
                }
            } else if coeff < 0 {
                if coeff == -1 {
                    parts.push(format!("-χ_{}", i));
                } else {
                    parts.push(format!("{}·χ_{}", coeff, i));
                }
            }
        }
        if parts.is_empty() { "0".to_string() } else { parts.join(" + ") }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// PermutationRepresentation — specialized for memory model symmetries
// ═══════════════════════════════════════════════════════════════════════

/// Specialized representation for memory model symmetry analysis.
///
/// Given an execution graph with n events and a symmetry group acting on it,
/// compute the representation and decompose it to identify invariant subspaces.
#[derive(Debug)]
pub struct PermutationRepresentationAnalysis;

impl PermutationRepresentationAnalysis {
    /// Analyze the representation arising from a symmetry group acting on events.
    pub fn analyze(
        group: &PermutationGroup,
        num_events: usize,
    ) -> PermutationAnalysisResult {
        let classes = compute_conjugacy_classes(group);
        let table = CharacterTable::compute(group);

        // Natural permutation representation character
        let nat_rep = GroupRepresentation::<Complex>::natural_representation(group);
        let chi_nat = Character::from_representation(&nat_rep, &classes);

        // Decompose
        let decomposition = IrreducibleDecomposition::decompose(&chi_nat, &table);

        // Count orbits on events
        let num_orbits = BurnsideCounter::count_point_orbits(group);

        // Count orbits on pairs (for relation compression)
        let num_pair_orbits = BurnsideCounter::count_orbits(group, |g| {
            let mut count = 0;
            for i in 0..group.degree() as u32 {
                for j in 0..group.degree() as u32 {
                    if g.apply(i) == i && g.apply(j) == j {
                        count += 1;
                    }
                }
            }
            count
        });

        PermutationAnalysisResult {
            group_order: group.order(),
            num_events,
            num_orbits,
            num_pair_orbits,
            character_table: table,
            decomposition,
            compression_ratio: num_events as f64 / num_orbits as f64,
        }
    }
}

/// Result of permutation representation analysis.
#[derive(Debug)]
pub struct PermutationAnalysisResult {
    /// Order of the symmetry group.
    pub group_order: u64,
    /// Number of events.
    pub num_events: usize,
    /// Number of event orbits.
    pub num_orbits: usize,
    /// Number of pair orbits (for relation matrices).
    pub num_pair_orbits: usize,
    /// Character table.
    pub character_table: CharacterTable,
    /// Decomposition of the permutation representation.
    pub decomposition: IrreducibleDecomposition,
    /// Compression ratio (events / orbits).
    pub compression_ratio: f64,
}

impl fmt::Display for PermutationAnalysisResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Permutation Representation Analysis")?;
        writeln!(f, "  Group order: {}", self.group_order)?;
        writeln!(f, "  Events: {}", self.num_events)?;
        writeln!(f, "  Event orbits: {}", self.num_orbits)?;
        writeln!(f, "  Pair orbits: {}", self.num_pair_orbits)?;
        writeln!(f, "  Compression ratio: {:.2}×", self.compression_ratio)?;
        writeln!(f, "  Decomposition: {}", self.decomposition.display())?;
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: create S3 (symmetric group on 3 elements)
    fn s3() -> PermutationGroup {
        let gen1 = Permutation::new(vec![1, 0, 2]); // (0 1)
        let gen2 = Permutation::new(vec![0, 2, 1]); // (1 2)
        PermutationGroup::new(3, vec![gen1, gen2])
    }

    // Helper: create Z4 (cyclic group of order 4)
    fn z4() -> PermutationGroup {
        let gen = Permutation::new(vec![1, 2, 3, 0]); // (0 1 2 3)
        PermutationGroup::new(4, vec![gen])
    }

    // Helper: create Z3 (cyclic group of order 3)
    fn z3() -> PermutationGroup {
        let gen = Permutation::new(vec![1, 2, 0]); // (0 1 2)
        PermutationGroup::new(3, vec![gen])
    }

    // ── Complex tests ────────────────────────────────────────────────

    #[test]
    fn test_complex_arithmetic() {
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(3.0, 4.0);

        let sum = a + b;
        assert!((sum.re - 4.0).abs() < 1e-10);
        assert!((sum.im - 6.0).abs() < 1e-10);

        let prod = a * b;
        assert!((prod.re - (-5.0)).abs() < 1e-10);
        assert!((prod.im - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_complex_conj_norm() {
        let z = Complex::new(3.0, 4.0);
        let conj = z.conj();
        assert!((conj.re - 3.0).abs() < 1e-10);
        assert!((conj.im - (-4.0)).abs() < 1e-10);
        assert!((z.norm() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_complex_inv() {
        let z = Complex::new(1.0, 1.0);
        let inv = z.inv();
        let product = z * inv;
        assert!(product.approx_eq(Complex::one(), 1e-10));
    }

    #[test]
    fn test_complex_root_of_unity() {
        let w = Complex::root_of_unity(4);
        assert!(w.approx_eq(Complex::new(0.0, 1.0), 1e-10));

        let w4 = w.powi(4);
        assert!(w4.approx_eq(Complex::one(), 1e-10));
    }

    #[test]
    fn test_complex_polar() {
        let z = Complex::from_polar(2.0, PI / 3.0);
        assert!((z.norm() - 2.0).abs() < 1e-10);
        assert!((z.arg() - PI / 3.0).abs() < 1e-10);
    }

    // ── Matrix tests ─────────────────────────────────────────────────

    #[test]
    fn test_matrix_identity() {
        let id = Matrix::<f64>::identity(3);
        assert_eq!(id.trace(), 3.0);
        assert_eq!(id.determinant(), 1.0);
    }

    #[test]
    fn test_matrix_mul() {
        let a = Matrix::from_rows(&[
            vec![1.0f64, 2.0],
            vec![3.0, 4.0],
        ]);
        let b = Matrix::from_rows(&[
            vec![5.0f64, 6.0],
            vec![7.0, 8.0],
        ]);
        let c = a.mul(&b);
        assert!((c.get(0, 0) - 19.0).abs() < 1e-10);
        assert!((c.get(0, 1) - 22.0).abs() < 1e-10);
        assert!((c.get(1, 0) - 43.0).abs() < 1e-10);
        assert!((c.get(1, 1) - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_determinant() {
        let m = Matrix::from_rows(&[
            vec![1.0f64, 2.0],
            vec![3.0, 4.0],
        ]);
        assert!((m.determinant() - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_inverse() {
        let m = Matrix::from_rows(&[
            vec![1.0f64, 2.0],
            vec![3.0, 4.0],
        ]);
        let inv = m.inverse().unwrap();
        let product = m.mul(&inv);
        assert!(product.approx_eq(&Matrix::identity(2), 1e-8));
    }

    #[test]
    fn test_matrix_transpose() {
        let m = Matrix::from_rows(&[
            vec![1.0f64, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ]);
        let t = m.transpose();
        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 2);
        assert!((t.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((t.get(2, 1) - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_direct_sum() {
        let a = Matrix::<f64>::identity(2);
        let b = Matrix::<f64>::identity(3);
        let ds = a.direct_sum(&b);
        assert_eq!(ds.rows, 5);
        assert_eq!(ds.cols, 5);
        assert!((ds.trace() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_kronecker() {
        let a = Matrix::from_rows(&[
            vec![1.0f64, 0.0],
            vec![0.0, 2.0],
        ]);
        let b = Matrix::<f64>::identity(2);
        let k = a.kronecker(&b);
        assert_eq!(k.rows, 4);
        assert_eq!(k.cols, 4);
        assert!((k.trace() - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_from_permutation() {
        let p = Permutation::new(vec![1, 0, 2]);
        let m = Matrix::<f64>::from_permutation(&p);
        assert!((m.get(0, 1) - 1.0).abs() < 1e-10);
        assert!((m.get(1, 0) - 1.0).abs() < 1e-10);
        assert!((m.get(2, 2) - 1.0).abs() < 1e-10);
        assert!((m.determinant() - (-1.0)).abs() < 1e-10);
    }

    // ── Conjugacy class tests ────────────────────────────────────────

    #[test]
    fn test_conjugacy_classes_s3() {
        let g = s3();
        let classes = compute_conjugacy_classes(&g);
        // S3 has 3 conjugacy classes: {e}, {3 transpositions}, {2 3-cycles}
        assert_eq!(classes.len(), 3);
        let sizes: Vec<usize> = {
            let mut s: Vec<usize> = classes.iter().map(|c| c.size()).collect();
            s.sort();
            s
        };
        assert_eq!(sizes, vec![1, 2, 3]);
    }

    #[test]
    fn test_conjugacy_classes_z4() {
        let g = z4();
        let classes = compute_conjugacy_classes(&g);
        // Z4 is abelian, so each element is its own conjugacy class
        assert_eq!(classes.len(), 4);
    }

    // ── Representation tests ─────────────────────────────────────────

    #[test]
    fn test_trivial_representation() {
        let g = s3();
        let rep = GroupRepresentation::<Complex>::trivial_representation(&g);
        assert_eq!(rep.dimension, 1);
        assert!(rep.is_valid(&g));
    }

    #[test]
    fn test_sign_representation() {
        let g = s3();
        let rep = GroupRepresentation::<Complex>::sign_representation(&g);
        assert_eq!(rep.dimension, 1);
        assert!(rep.is_valid(&g));
    }

    #[test]
    fn test_natural_representation() {
        let g = s3();
        let rep = GroupRepresentation::<Complex>::natural_representation(&g);
        assert_eq!(rep.dimension, 3);
        assert!(rep.is_valid(&g));
    }

    #[test]
    fn test_regular_representation() {
        let g = z3();
        let rep = GroupRepresentation::<Complex>::regular_representation(&g);
        assert_eq!(rep.dimension, 3);
        assert!(rep.is_valid(&g));
    }

    #[test]
    fn test_direct_sum_representation() {
        let g = s3();
        let triv = GroupRepresentation::<Complex>::trivial_representation(&g);
        let sign = GroupRepresentation::<Complex>::sign_representation(&g);
        let ds = triv.direct_sum(&sign);
        assert_eq!(ds.dimension, 2);
        assert!(ds.is_valid(&g));
    }

    #[test]
    fn test_tensor_product_representation() {
        let g = s3();
        let triv = GroupRepresentation::<Complex>::trivial_representation(&g);
        let sign = GroupRepresentation::<Complex>::sign_representation(&g);
        let tp = triv.tensor_product(&sign);
        assert_eq!(tp.dimension, 1);
        assert!(tp.is_valid(&g));
    }

    // ── Character tests ──────────────────────────────────────────────

    #[test]
    fn test_character_trivial() {
        let g = s3();
        let classes = compute_conjugacy_classes(&g);
        let rep = GroupRepresentation::<Complex>::trivial_representation(&g);
        let chi = Character::from_representation(&rep, &classes);

        // Trivial character is 1 on all elements
        for (_, &val) in &chi.values {
            assert!(val.approx_eq(Complex::one(), 1e-10));
        }
    }

    #[test]
    fn test_character_sign_s3() {
        let g = s3();
        let classes = compute_conjugacy_classes(&g);
        let rep = GroupRepresentation::<Complex>::sign_representation(&g);
        let chi = Character::from_representation(&rep, &classes);

        assert!(chi.is_irreducible(&classes, g.order()));
    }

    #[test]
    fn test_character_inner_product() {
        let g = s3();
        let classes = compute_conjugacy_classes(&g);
        let triv = GroupRepresentation::<Complex>::trivial_representation(&g);
        let sign = GroupRepresentation::<Complex>::sign_representation(&g);
        let chi_triv = Character::from_representation(&triv, &classes);
        let chi_sign = Character::from_representation(&sign, &classes);

        // ⟨χ_triv, χ_triv⟩ = 1
        let ip11 = chi_triv.inner_product(&chi_triv, &classes, g.order());
        assert!((ip11.abs() - 1.0).abs() < 0.01);

        // ⟨χ_triv, χ_sign⟩ = 0
        let ip12 = chi_triv.inner_product(&chi_sign, &classes, g.order());
        assert!(ip12.abs() < 0.01);
    }

    // ── Character table tests ────────────────────────────────────────

    #[test]
    fn test_character_table_s3() {
        let g = s3();
        let table = CharacterTable::compute(&g);

        assert_eq!(table.num_classes(), 3);
        assert_eq!(table.group_order, 6);
    }

    #[test]
    fn test_character_table_z4() {
        let g = z4();
        let table = CharacterTable::compute(&g);

        assert_eq!(table.num_classes(), 4);
        assert_eq!(table.group_order, 4);
    }

    // ── Burnside tests ───────────────────────────────────────────────

    #[test]
    fn test_burnside_point_orbits_s3() {
        let g = s3();
        // S3 acts transitively on {0,1,2}, so 1 orbit
        let orbits = BurnsideCounter::count_point_orbits(&g);
        assert_eq!(orbits, 1);
    }

    #[test]
    fn test_burnside_point_orbits_z4() {
        let g = z4();
        // Z4 = ⟨(0123)⟩ acts transitively on {0,1,2,3}, so 1 orbit
        let orbits = BurnsideCounter::count_point_orbits(&g);
        assert_eq!(orbits, 1);
    }

    #[test]
    fn test_burnside_colorings() {
        // Necklaces with 3 beads and 2 colors under Z3
        let g = z3();
        let count = BurnsideCounter::count_colorings(&g, 2);
        // (2^3 + 2 + 2) / 3 = 12/3 = 4
        assert_eq!(count, 4);
    }

    #[test]
    fn test_burnside_subset_orbits() {
        let g = s3();
        // 1-element subsets of {0,1,2} under S3: 1 orbit
        let orbits1 = BurnsideCounter::count_subset_orbits(&g, 1);
        assert_eq!(orbits1, 1);

        // 2-element subsets: also 1 orbit
        let orbits2 = BurnsideCounter::count_subset_orbits(&g, 2);
        assert_eq!(orbits2, 1);
    }

    // ── Schur orthogonality tests ────────────────────────────────────

    #[test]
    fn test_dimension_formula_s3() {
        let g = s3();
        let table = CharacterTable::compute(&g);
        assert!(SchurOrthogonality::verify_dimension_formula(&table));
    }

    // ── Decomposition tests ──────────────────────────────────────────

    #[test]
    fn test_decompose_natural_representation_s3() {
        let g = s3();
        let table = CharacterTable::compute(&g);
        let classes = compute_conjugacy_classes(&g);
        let nat = GroupRepresentation::<Complex>::natural_representation(&g);
        let chi_nat = Character::from_representation(&nat, &classes);

        let decomp = IrreducibleDecomposition::decompose(&chi_nat, &table);
        assert_eq!(decomp.total_dimension, 3);
        assert!(!decomp.is_irreducible());
    }

    // ── Fourier transform tests ──────────────────────────────────────

    #[test]
    fn test_fourier_constant_function() {
        let g = z3();
        let elements = g.elements();
        let ft = FourierTransform::new(g.order());

        // Constant function f(g) = 1 for all g
        let mut f = HashMap::new();
        for elem in &elements {
            let key: Vec<u32> = (0..elem.degree() as u32).map(|x| elem.apply(x)).collect();
            f.insert(key, Complex::one());
        }

        let trivial = GroupRepresentation::<Complex>::trivial_representation(&g);
        let transforms = ft.forward(&f, &[trivial]);

        // For constant function, Fourier transform at trivial rep = |G|
        assert_eq!(transforms.len(), 1);
        let val = transforms[0].get(0, 0);
        assert!((val.re - 3.0).abs() < 1e-8);
    }

    // ── Tensor product tests ─────────────────────────────────────────

    #[test]
    fn test_tensor_product_characters() {
        let g = s3();
        let classes = compute_conjugacy_classes(&g);
        let triv = GroupRepresentation::<Complex>::trivial_representation(&g);
        let sign = GroupRepresentation::<Complex>::sign_representation(&g);
        let chi_triv = Character::from_representation(&triv, &classes);
        let chi_sign = Character::from_representation(&sign, &classes);

        // trivial ⊗ sign = sign
        let product = TensorProduct::character(&chi_triv, &chi_sign);
        assert_eq!(product.dimension, 1);
    }

    // ── Representation ring tests ────────────────────────────────────

    #[test]
    fn test_representation_ring_addition() {
        let g = s3();
        let table = CharacterTable::compute(&g);
        let ring = RepresentationRing::new(table);

        let e0 = ring.irreducible(0);
        let e1 = ring.irreducible(1);
        let sum = ring.add(&e0, &e1);

        assert_eq!(sum.coefficients[0], 1);
        assert_eq!(sum.coefficients[1], 1);
    }

    #[test]
    fn test_representation_ring_zero() {
        let g = s3();
        let table = CharacterTable::compute(&g);
        let ring = RepresentationRing::new(table);

        let zero = ring.zero();
        assert_eq!(ring.dimension(&zero), 0);
    }

    // ── Permutation analysis tests ───────────────────────────────────

    #[test]
    fn test_permutation_analysis_s3() {
        let g = s3();
        let result = PermutationRepresentationAnalysis::analyze(&g, 3);
        assert_eq!(result.group_order, 6);
        assert_eq!(result.num_events, 3);
        assert_eq!(result.num_orbits, 1);
        assert!(result.compression_ratio > 1.0);
    }

    #[test]
    fn test_permutation_analysis_z4() {
        let g = z4();
        let result = PermutationRepresentationAnalysis::analyze(&g, 4);
        assert_eq!(result.group_order, 4);
        assert_eq!(result.num_orbits, 1);
    }
}
