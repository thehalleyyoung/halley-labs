#![allow(unused)]
//! Matrix representation of groups for LITMUS∞ algebraic engine.
//!
//! Implements dense matrix operations, group representations, character theory,
//! character polynomials, matrix group operations, and the representation ring.

use std::collections::{HashMap, HashSet, BTreeMap};
use std::fmt;
use std::ops;

// =========================================================================
// Complex Numbers
// =========================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub fn new(re: f64, im: f64) -> Self { Complex { re, im } }
    pub fn zero() -> Self { Complex { re: 0.0, im: 0.0 } }
    pub fn one() -> Self { Complex { re: 1.0, im: 0.0 } }
    pub fn i() -> Self { Complex { re: 0.0, im: 1.0 } }
    pub fn conjugate(&self) -> Self { Complex { re: self.re, im: -self.im } }
    pub fn norm_sq(&self) -> f64 { self.re * self.re + self.im * self.im }
    pub fn norm(&self) -> f64 { self.norm_sq().sqrt() }
    pub fn arg(&self) -> f64 { self.im.atan2(self.re) }
    pub fn from_polar(r: f64, theta: f64) -> Self {
        Complex { re: r * theta.cos(), im: r * theta.sin() }
    }
    pub fn exp(&self) -> Self {
        let r = self.re.exp();
        Complex { re: r * self.im.cos(), im: r * self.im.sin() }
    }
    pub fn is_real(&self) -> bool { self.im.abs() < 1e-12 }
    pub fn is_zero(&self) -> bool { self.norm_sq() < 1e-24 }
}

impl fmt::Display for Complex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.im.abs() < 1e-12 { write!(f, "{:.4}", self.re) }
        else if self.re.abs() < 1e-12 { write!(f, "{:.4}i", self.im) }
        else if self.im > 0.0 { write!(f, "{:.4}+{:.4}i", self.re, self.im) }
        else { write!(f, "{:.4}{:.4}i", self.re, self.im) }
    }
}

impl ops::Add for Complex {
    type Output = Self;
    fn add(self, rhs: Self) -> Self { Complex { re: self.re + rhs.re, im: self.im + rhs.im } }
}

impl ops::Sub for Complex {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self { Complex { re: self.re - rhs.re, im: self.im - rhs.im } }
}

impl ops::Mul for Complex {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Complex {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl ops::Div for Complex {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        let d = rhs.norm_sq();
        Complex {
            re: (self.re * rhs.re + self.im * rhs.im) / d,
            im: (self.im * rhs.re - self.re * rhs.im) / d,
        }
    }
}

impl ops::Neg for Complex {
    type Output = Self;
    fn neg(self) -> Self { Complex { re: -self.re, im: -self.im } }
}

impl ops::Mul<f64> for Complex {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self { Complex { re: self.re * rhs, im: self.im * rhs } }
}

impl ops::AddAssign for Complex {
    fn add_assign(&mut self, rhs: Self) { self.re += rhs.re; self.im += rhs.im; }
}

impl ops::SubAssign for Complex {
    fn sub_assign(&mut self, rhs: Self) { self.re -= rhs.re; self.im -= rhs.im; }
}

impl ops::MulAssign for Complex {
    fn mul_assign(&mut self, rhs: Self) {
        let re = self.re * rhs.re - self.im * rhs.im;
        let im = self.re * rhs.im + self.im * rhs.re;
        self.re = re; self.im = im;
    }
}

// =========================================================================
// Dense Matrix
// =========================================================================

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<f64>>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Matrix { rows, cols, data: vec![vec![0.0; cols]; rows] }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::new(n, n);
        for i in 0..n { m.data[i][i] = 1.0; }
        m
    }

    pub fn zero(rows: usize, cols: usize) -> Self { Self::new(rows, cols) }

    pub fn from_rows(data: Vec<Vec<f64>>) -> Self {
        let rows = data.len();
        let cols = if rows > 0 { data[0].len() } else { 0 };
        Matrix { rows, cols, data }
    }

    pub fn get(&self, i: usize, j: usize) -> f64 { self.data[i][j] }
    pub fn set(&mut self, i: usize, j: usize, v: f64) { self.data[i][j] = v; }

    pub fn transpose(&self) -> Self {
        let mut m = Matrix::new(self.cols, self.rows);
        for i in 0..self.rows { for j in 0..self.cols { m.data[j][i] = self.data[i][j]; } }
        m
    }

    pub fn multiply(&self, other: &Self) -> Self {
        assert_eq!(self.cols, other.rows);
        let mut m = Matrix::new(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut s = 0.0;
                for k in 0..self.cols { s += self.data[i][k] * other.data[k][j]; }
                m.data[i][j] = s;
            }
        }
        m
    }

    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let mut m = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols { m.data[i][j] = self.data[i][j] + other.data[i][j]; }
        }
        m
    }

    pub fn sub(&self, other: &Self) -> Self {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let mut m = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols { m.data[i][j] = self.data[i][j] - other.data[i][j]; }
        }
        m
    }

    pub fn scalar_mul(&self, s: f64) -> Self {
        let mut m = self.clone();
        for i in 0..self.rows { for j in 0..self.cols { m.data[i][j] *= s; } }
        m
    }

    pub fn trace(&self) -> f64 {
        assert_eq!(self.rows, self.cols);
        (0..self.rows).map(|i| self.data[i][i]).sum()
    }

    pub fn determinant(&self) -> f64 {
        assert_eq!(self.rows, self.cols);
        let n = self.rows;
        if n == 0 { return 1.0; }
        if n == 1 { return self.data[0][0]; }
        if n == 2 { return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]; }
        // LU decomposition based
        let mut a = self.data.clone();
        let mut det = 1.0;
        for col in 0..n {
            // Find pivot
            let mut pivot = col;
            for row in (col+1)..n {
                if a[row][col].abs() > a[pivot][col].abs() { pivot = row; }
            }
            if a[pivot][col].abs() < 1e-15 { return 0.0; }
            if pivot != col { a.swap(col, pivot); det = -det; }
            det *= a[col][col];
            for row in (col+1)..n {
                let factor = a[row][col] / a[col][col];
                for j in col..n { a[row][j] -= factor * a[col][j]; }
            }
        }
        det
    }

    pub fn rank(&self) -> usize {
        let mut a = self.data.clone();
        let (m, n) = (self.rows, self.cols);
        let mut r = 0;
        for col in 0..n {
            let mut pivot = None;
            for row in r..m {
                if a[row][col].abs() > 1e-12 { pivot = Some(row); break; }
            }
            let pivot = match pivot { Some(p) => p, None => continue };
            a.swap(r, pivot);
            let scale = a[r][col];
            for j in 0..n { a[r][j] /= scale; }
            for row in 0..m {
                if row == r { continue; }
                let factor = a[row][col];
                for j in 0..n { a[row][j] -= factor * a[r][j]; }
            }
            r += 1;
        }
        r
    }

    pub fn inverse(&self) -> Option<Self> {
        assert_eq!(self.rows, self.cols);
        let n = self.rows;
        let mut aug = Matrix::new(n, 2 * n);
        for i in 0..n {
            for j in 0..n { aug.data[i][j] = self.data[i][j]; }
            aug.data[i][n + i] = 1.0;
        }
        for col in 0..n {
            let mut pivot = col;
            for row in (col+1)..n {
                if aug.data[row][col].abs() > aug.data[pivot][col].abs() { pivot = row; }
            }
            if aug.data[pivot][col].abs() < 1e-15 { return None; }
            aug.data.swap(col, pivot);
            let scale = aug.data[col][col];
            for j in 0..(2*n) { aug.data[col][j] /= scale; }
            for row in 0..n {
                if row == col { continue; }
                let factor = aug.data[row][col];
                for j in 0..(2*n) { aug.data[row][j] -= factor * aug.data[col][j]; }
            }
        }
        let mut inv = Matrix::new(n, n);
        for i in 0..n { for j in 0..n { inv.data[i][j] = aug.data[i][n + j]; } }
        Some(inv)
    }

    pub fn kronecker(&self, other: &Self) -> Self {
        let (m1, n1) = (self.rows, self.cols);
        let (m2, n2) = (other.rows, other.cols);
        let mut result = Matrix::new(m1 * m2, n1 * n2);
        for i1 in 0..m1 {
            for j1 in 0..n1 {
                for i2 in 0..m2 {
                    for j2 in 0..n2 {
                        result.data[i1 * m2 + i2][j1 * n2 + j2] = self.data[i1][j1] * other.data[i2][j2];
                    }
                }
            }
        }
        result
    }

    pub fn direct_sum(&self, other: &Self) -> Self {
        let (m1, n1) = (self.rows, self.cols);
        let (m2, n2) = (other.rows, other.cols);
        let mut result = Matrix::new(m1 + m2, n1 + n2);
        for i in 0..m1 { for j in 0..n1 { result.data[i][j] = self.data[i][j]; } }
        for i in 0..m2 { for j in 0..n2 { result.data[m1 + i][n1 + j] = other.data[i][j]; } }
        result
    }

    pub fn frobenius_norm(&self) -> f64 {
        let mut s = 0.0;
        for row in &self.data { for &v in row { s += v * v; } }
        s.sqrt()
    }

    pub fn is_symmetric(&self) -> bool {
        if self.rows != self.cols { return false; }
        for i in 0..self.rows { for j in (i+1)..self.cols {
            if (self.data[i][j] - self.data[j][i]).abs() > 1e-12 { return false; }
        }}
        true
    }

    pub fn is_orthogonal(&self) -> bool {
        if self.rows != self.cols { return false; }
        let product = self.multiply(&self.transpose());
        let id = Matrix::identity(self.rows);
        product.sub(&id).frobenius_norm() < 1e-10
    }

    pub fn characteristic_polynomial(&self) -> Polynomial {
        assert_eq!(self.rows, self.cols);
        let n = self.rows;
        if n == 0 { return Polynomial::new(vec![1.0]); }
        if n == 1 { return Polynomial::new(vec![-self.data[0][0], 1.0]); }
        if n == 2 {
            let a = self.data[0][0];
            let b = self.data[0][1];
            let c = self.data[1][0];
            let d = self.data[1][1];
            // λ² - (a+d)λ + (ad-bc)
            return Polynomial::new(vec![a*d - b*c, -(a + d), 1.0]);
        }
        // Faddeev-LeVerrier algorithm
        let mut c = vec![0.0; n + 1];
        c[n] = 1.0;
        let mut m = Matrix::identity(n);
        for k in 1..=n {
            m = self.multiply(&m);
            c[n - k] = -m.trace() / k as f64;
            if k < n {
                for i in 0..n { m.data[i][i] += c[n - k]; }
            }
        }
        Polynomial::new(c)
    }

    pub fn eigenvalues_2x2(&self) -> Option<(Complex, Complex)> {
        if self.rows != 2 || self.cols != 2 { return None; }
        let a = self.data[0][0];
        let b = self.data[0][1];
        let c = self.data[1][0];
        let d = self.data[1][1];
        let tr = a + d;
        let det = a * d - b * c;
        let disc = tr * tr - 4.0 * det;
        if disc >= 0.0 {
            let sqrt_disc = disc.sqrt();
            Some((
                Complex::new((tr + sqrt_disc) / 2.0, 0.0),
                Complex::new((tr - sqrt_disc) / 2.0, 0.0),
            ))
        } else {
            let sqrt_disc = (-disc).sqrt();
            Some((
                Complex::new(tr / 2.0, sqrt_disc / 2.0),
                Complex::new(tr / 2.0, -sqrt_disc / 2.0),
            ))
        }
    }

    pub fn power(&self, k: u32) -> Self {
        assert_eq!(self.rows, self.cols);
        if k == 0 { return Matrix::identity(self.rows); }
        let mut result = Matrix::identity(self.rows);
        let mut base = self.clone();
        let mut exp = k;
        while exp > 0 {
            if exp % 2 == 1 { result = result.multiply(&base); }
            base = base.multiply(&base);
            exp /= 2;
        }
        result
    }

    pub fn row_echelon(&self) -> Self {
        let mut a = self.clone();
        let mut r = 0;
        for col in 0..a.cols {
            let mut pivot = None;
            for row in r..a.rows {
                if a.data[row][col].abs() > 1e-12 { pivot = Some(row); break; }
            }
            let pivot = match pivot { Some(p) => p, None => continue };
            a.data.swap(r, pivot);
            let scale = a.data[r][col];
            for j in 0..a.cols { a.data[r][j] /= scale; }
            for row in (r+1)..a.rows {
                let factor = a.data[row][col];
                for j in 0..a.cols { a.data[row][j] -= factor * a.data[r][j]; }
            }
            r += 1;
        }
        a
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, row) in self.data.iter().enumerate() {
            write!(f, "[")?;
            for (j, &v) in row.iter().enumerate() {
                if j > 0 { write!(f, ", ")?; }
                write!(f, "{:8.4}", v)?;
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}

// =========================================================================
// Complex Matrix
// =========================================================================

#[derive(Debug, Clone, PartialEq)]
pub struct ComplexMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<Complex>>,
}

impl ComplexMatrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        ComplexMatrix { rows, cols, data: vec![vec![Complex::zero(); cols]; rows] }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::new(n, n);
        for i in 0..n { m.data[i][i] = Complex::one(); }
        m
    }

    pub fn from_real(m: &Matrix) -> Self {
        let mut cm = ComplexMatrix::new(m.rows, m.cols);
        for i in 0..m.rows { for j in 0..m.cols { cm.data[i][j] = Complex::new(m.data[i][j], 0.0); } }
        cm
    }

    pub fn multiply(&self, other: &Self) -> Self {
        assert_eq!(self.cols, other.rows);
        let mut m = ComplexMatrix::new(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut s = Complex::zero();
                for k in 0..self.cols { s += self.data[i][k] * other.data[k][j]; }
                m.data[i][j] = s;
            }
        }
        m
    }

    pub fn trace(&self) -> Complex {
        assert_eq!(self.rows, self.cols);
        let mut t = Complex::zero();
        for i in 0..self.rows { t += self.data[i][i]; }
        t
    }

    pub fn conjugate_transpose(&self) -> Self {
        let mut m = ComplexMatrix::new(self.cols, self.rows);
        for i in 0..self.rows { for j in 0..self.cols { m.data[j][i] = self.data[i][j].conjugate(); } }
        m
    }

    pub fn is_unitary(&self) -> bool {
        if self.rows != self.cols { return false; }
        let product = self.multiply(&self.conjugate_transpose());
        let id = ComplexMatrix::identity(self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                if (product.data[i][j] - id.data[i][j]).norm() > 1e-10 { return false; }
            }
        }
        true
    }

    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let mut m = ComplexMatrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols { m.data[i][j] = self.data[i][j] + other.data[i][j]; }
        }
        m
    }

    pub fn scalar_mul(&self, s: Complex) -> Self {
        let mut m = self.clone();
        for i in 0..self.rows { for j in 0..self.cols { m.data[i][j] = m.data[i][j] * s; } }
        m
    }

    pub fn kronecker(&self, other: &Self) -> Self {
        let (m1, n1) = (self.rows, self.cols);
        let (m2, n2) = (other.rows, other.cols);
        let mut result = ComplexMatrix::new(m1 * m2, n1 * n2);
        for i1 in 0..m1 {
            for j1 in 0..n1 {
                for i2 in 0..m2 {
                    for j2 in 0..n2 {
                        result.data[i1*m2+i2][j1*n2+j2] = self.data[i1][j1] * other.data[i2][j2];
                    }
                }
            }
        }
        result
    }
}

// =========================================================================
// Polynomial
// =========================================================================

#[derive(Debug, Clone, PartialEq)]
pub struct Polynomial {
    pub coeffs: Vec<f64>,
}

impl Polynomial {
    pub fn new(coeffs: Vec<f64>) -> Self { Polynomial { coeffs } }
    pub fn zero() -> Self { Polynomial { coeffs: vec![0.0] } }
    pub fn one() -> Self { Polynomial { coeffs: vec![1.0] } }
    pub fn monomial(degree: usize, coeff: f64) -> Self {
        let mut c = vec![0.0; degree + 1];
        c[degree] = coeff;
        Polynomial { coeffs: c }
    }

    pub fn degree(&self) -> usize {
        for i in (0..self.coeffs.len()).rev() {
            if self.coeffs[i].abs() > 1e-15 { return i; }
        }
        0
    }

    pub fn evaluate(&self, x: f64) -> f64 {
        let mut result = 0.0;
        let mut xp = 1.0;
        for &c in &self.coeffs { result += c * xp; xp *= x; }
        result
    }

    pub fn evaluate_complex(&self, z: Complex) -> Complex {
        let mut result = Complex::zero();
        let mut zp = Complex::one();
        for &c in &self.coeffs { result += zp * c; zp *= z; }
        result
    }

    pub fn add(&self, other: &Self) -> Self {
        let n = self.coeffs.len().max(other.coeffs.len());
        let mut c = vec![0.0; n];
        for i in 0..self.coeffs.len() { c[i] += self.coeffs[i]; }
        for i in 0..other.coeffs.len() { c[i] += other.coeffs[i]; }
        Polynomial { coeffs: c }
    }

    pub fn mul(&self, other: &Self) -> Self {
        if self.coeffs.is_empty() || other.coeffs.is_empty() {
            return Polynomial::zero();
        }
        let n = self.coeffs.len() + other.coeffs.len() - 1;
        let mut c = vec![0.0; n];
        for i in 0..self.coeffs.len() {
            for j in 0..other.coeffs.len() { c[i + j] += self.coeffs[i] * other.coeffs[j]; }
        }
        Polynomial { coeffs: c }
    }

    pub fn scalar_mul(&self, s: f64) -> Self {
        Polynomial { coeffs: self.coeffs.iter().map(|&c| c * s).collect() }
    }

    pub fn derivative(&self) -> Self {
        if self.coeffs.len() <= 1 { return Polynomial::zero(); }
        let c: Vec<f64> = (1..self.coeffs.len())
            .map(|i| self.coeffs[i] * i as f64)
            .collect();
        Polynomial { coeffs: c }
    }

    pub fn roots_quadratic(&self) -> Option<(Complex, Complex)> {
        if self.degree() != 2 { return None; }
        let a = self.coeffs[2];
        let b = self.coeffs[1];
        let c = self.coeffs[0];
        let disc = b * b - 4.0 * a * c;
        if disc >= 0.0 {
            let sq = disc.sqrt();
            Some((Complex::new((-b + sq) / (2.0 * a), 0.0),
                  Complex::new((-b - sq) / (2.0 * a), 0.0)))
        } else {
            let sq = (-disc).sqrt();
            Some((Complex::new(-b / (2.0 * a), sq / (2.0 * a)),
                  Complex::new(-b / (2.0 * a), -sq / (2.0 * a))))
        }
    }
}

impl fmt::Display for Polynomial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        for (i, &c) in self.coeffs.iter().enumerate().rev() {
            if c.abs() < 1e-15 { continue; }
            if !first && c > 0.0 { write!(f, " + ")?; }
            else if !first && c < 0.0 { write!(f, " - ")?; }
            else if c < 0.0 { write!(f, "-")?; }
            let ac = c.abs();
            match i {
                0 => write!(f, "{:.4}", ac)?,
                1 => { if (ac - 1.0).abs() > 1e-12 { write!(f, "{:.4}", ac)?; } write!(f, "x")?; }
                _ => { if (ac - 1.0).abs() > 1e-12 { write!(f, "{:.4}", ac)?; } write!(f, "x^{}", i)?; }
            }
            first = false;
        }
        if first { write!(f, "0")?; }
        Ok(())
    }
}

// =========================================================================
// Group Representation
// =========================================================================

#[derive(Debug, Clone)]
pub struct Representation {
    pub dimension: usize,
    pub group_order: usize,
    pub matrices: HashMap<usize, Matrix>,
    pub element_names: HashMap<usize, String>,
}

impl Representation {
    pub fn new(dimension: usize) -> Self {
        Representation {
            dimension, group_order: 0,
            matrices: HashMap::new(), element_names: HashMap::new(),
        }
    }

    pub fn add_element(&mut self, id: usize, name: &str, matrix: Matrix) {
        assert_eq!(matrix.rows, self.dimension);
        assert_eq!(matrix.cols, self.dimension);
        self.matrices.insert(id, matrix);
        self.element_names.insert(id, name.to_string());
        self.group_order = self.matrices.len();
    }

    pub fn degree(&self) -> usize { self.dimension }

    pub fn check_homomorphism(&self, mult_table: &HashMap<(usize, usize), usize>) -> bool {
        for (&(g, h), &gh) in mult_table {
            let mg = match self.matrices.get(&g) { Some(m) => m, None => return false };
            let mh = match self.matrices.get(&h) { Some(m) => m, None => return false };
            let mgh = match self.matrices.get(&gh) { Some(m) => m, None => return false };
            let product = mg.multiply(mh);
            if product.sub(mgh).frobenius_norm() > 1e-10 { return false; }
        }
        true
    }

    pub fn is_faithful(&self) -> bool {
        let identity = Matrix::identity(self.dimension);
        let mut identity_count = 0;
        for (_, m) in &self.matrices {
            if m.sub(&identity).frobenius_norm() < 1e-10 { identity_count += 1; }
        }
        identity_count <= 1
    }

    pub fn kernel(&self) -> Vec<usize> {
        let identity = Matrix::identity(self.dimension);
        self.matrices.iter()
            .filter(|(_, m)| m.sub(&identity).frobenius_norm() < 1e-10)
            .map(|(&id, _)| id)
            .collect()
    }
}

// =========================================================================
// Character
// =========================================================================

#[derive(Debug, Clone)]
pub struct Character {
    pub values: HashMap<usize, Complex>,
    pub dimension: usize,
}

impl Character {
    pub fn from_representation(rep: &Representation) -> Self {
        let mut values = HashMap::new();
        for (&id, m) in &rep.matrices {
            values.insert(id, Complex::new(m.trace(), 0.0));
        }
        Character { values, dimension: rep.dimension }
    }

    pub fn inner_product(&self, other: &Character, group_order: usize) -> Complex {
        let mut sum = Complex::zero();
        for (&g, &chi1_g) in &self.values {
            if let Some(&chi2_g) = other.values.get(&g) {
                sum += chi1_g * chi2_g.conjugate();
            }
        }
        sum * (1.0 / group_order as f64)
    }

    pub fn is_irreducible(&self, group_order: usize) -> bool {
        let ip = self.inner_product(self, group_order);
        (ip.re - 1.0).abs() < 0.1 && ip.im.abs() < 0.1
    }

    pub fn norm_squared(&self, group_order: usize) -> f64 {
        self.inner_product(self, group_order).re
    }
}

// =========================================================================
// Character Table
// =========================================================================

#[derive(Debug, Clone)]
pub struct CharacterTable {
    pub group_order: usize,
    pub num_classes: usize,
    pub class_sizes: Vec<usize>,
    pub class_representatives: Vec<usize>,
    pub characters: Vec<Character>,
}

impl CharacterTable {
    pub fn new(group_order: usize) -> Self {
        CharacterTable {
            group_order, num_classes: 0,
            class_sizes: Vec::new(),
            class_representatives: Vec::new(),
            characters: Vec::new(),
        }
    }

    pub fn add_class(&mut self, representative: usize, size: usize) {
        self.class_representatives.push(representative);
        self.class_sizes.push(size);
        self.num_classes += 1;
    }

    pub fn add_character(&mut self, chi: Character) {
        self.characters.push(chi);
    }

    pub fn orthogonality_check(&self) -> bool {
        for i in 0..self.characters.len() {
            for j in 0..self.characters.len() {
                let ip = self.characters[i].inner_product(&self.characters[j], self.group_order);
                let expected = if i == j { 1.0 } else { 0.0 };
                if (ip.re - expected).abs() > 0.1 || ip.im.abs() > 0.1 { return false; }
            }
        }
        true
    }
}

impl fmt::Display for CharacterTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Character Table (|G| = {}):", self.group_order)?;
        write!(f, "     ")?;
        for &rep in &self.class_representatives { write!(f, " C{:3}", rep)?; }
        writeln!(f)?;
        for (i, chi) in self.characters.iter().enumerate() {
            write!(f, "χ_{:2} ", i)?;
            for &rep in &self.class_representatives {
                let v = chi.values.get(&rep).copied().unwrap_or(Complex::zero());
                write!(f, " {:4}", v)?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

// =========================================================================
// Character Decomposition
// =========================================================================

#[derive(Debug, Clone)]
pub struct CharacterDecomposition {
    pub multiplicities: Vec<usize>,
    pub irreducibles: Vec<usize>,
}

impl CharacterDecomposition {
    pub fn decompose(chi: &Character, table: &CharacterTable) -> Self {
        let mut multiplicities = Vec::new();
        let mut irreducibles = Vec::new();
        for (i, irr) in table.characters.iter().enumerate() {
            let ip = chi.inner_product(irr, table.group_order);
            let mult = ip.re.round() as usize;
            multiplicities.push(mult);
            if mult > 0 { irreducibles.push(i); }
        }
        CharacterDecomposition { multiplicities, irreducibles }
    }
}

impl fmt::Display for CharacterDecomposition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let parts: Vec<String> = self.multiplicities.iter().enumerate()
            .filter(|(_, &m)| m > 0)
            .map(|(i, &m)| if m == 1 { format!("χ_{}", i) } else { format!("{}·χ_{}", m, i) })
            .collect();
        write!(f, "{}", parts.join(" ⊕ "))
    }
}

// =========================================================================
// Molien Series
// =========================================================================

pub fn molien_series_terms(rep: &Representation, num_terms: usize) -> Vec<f64> {
    let n = rep.group_order;
    if n == 0 { return vec![0.0; num_terms]; }
    let d = rep.dimension;
    let id = Matrix::identity(d);
    let mut terms = vec![0.0; num_terms];
    // Approximate: use trace powers
    for (_, m) in &rep.matrices {
        let mut m_power = Matrix::identity(d);
        for k in 0..num_terms {
            terms[k] += m_power.trace();
            m_power = m_power.multiply(m);
        }
    }
    for t in &mut terms { *t /= n as f64; }
    terms
}

// =========================================================================
// Matrix Group
// =========================================================================

#[derive(Debug, Clone)]
pub struct MatrixGroup {
    pub generators: Vec<Matrix>,
    pub elements: Vec<Matrix>,
    pub dimension: usize,
}

impl MatrixGroup {
    pub fn from_generators(generators: Vec<Matrix>) -> Self {
        let dimension = if generators.is_empty() { 0 } else { generators[0].rows };
        let mut elements = vec![Matrix::identity(dimension)];
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(Matrix::identity(dimension));
        let max_order = 10000;
        while let Some(m) = queue.pop_front() {
            if elements.len() >= max_order { break; }
            for g in &generators {
                let product = m.multiply(g);
                let is_new = !elements.iter().any(|e| e.sub(&product).frobenius_norm() < 1e-10);
                if is_new {
                    elements.push(product.clone());
                    queue.push_back(product);
                }
            }
        }
        MatrixGroup { generators, elements, dimension }
    }

    pub fn order(&self) -> usize { self.elements.len() }

    pub fn center(&self) -> Vec<Matrix> {
        self.elements.iter().filter(|c| {
            self.elements.iter().all(|g| {
                let cg = c.multiply(g);
                let gc = g.multiply(c);
                cg.sub(&gc).frobenius_norm() < 1e-10
            })
        }).cloned().collect()
    }

    pub fn is_abelian(&self) -> bool {
        for i in 0..self.elements.len() {
            for j in (i+1)..self.elements.len() {
                let ab = self.elements[i].multiply(&self.elements[j]);
                let ba = self.elements[j].multiply(&self.elements[i]);
                if ab.sub(&ba).frobenius_norm() > 1e-10 { return false; }
            }
        }
        true
    }

    pub fn conjugacy_classes(&self) -> Vec<Vec<usize>> {
        let n = self.elements.len();
        let mut assigned = vec![false; n];
        let mut classes = Vec::new();
        for i in 0..n {
            if assigned[i] { continue; }
            let mut class = vec![i];
            assigned[i] = true;
            for j in (i+1)..n {
                if assigned[j] { continue; }
                let is_conjugate = self.elements.iter().any(|g| {
                    let ginv = g.inverse();
                    if let Some(gi) = ginv {
                        let conj = g.multiply(&self.elements[i]).multiply(&gi);
                        conj.sub(&self.elements[j]).frobenius_norm() < 1e-10
                    } else { false }
                });
                if is_conjugate { class.push(j); assigned[j] = true; }
            }
            classes.push(class);
        }
        classes
    }
}

// =========================================================================
// Representation Ring
// =========================================================================

#[derive(Debug, Clone)]
pub struct RingElement {
    pub coefficients: Vec<i64>,
}

impl RingElement {
    pub fn new(coefficients: Vec<i64>) -> Self { RingElement { coefficients } }
    pub fn zero(n: usize) -> Self { RingElement { coefficients: vec![0; n] } }

    pub fn add(&self, other: &Self) -> Self {
        let n = self.coefficients.len().max(other.coefficients.len());
        let mut c = vec![0i64; n];
        for (i, &v) in self.coefficients.iter().enumerate() { c[i] += v; }
        for (i, &v) in other.coefficients.iter().enumerate() { c[i] += v; }
        RingElement { coefficients: c }
    }

    pub fn sub(&self, other: &Self) -> Self {
        let n = self.coefficients.len().max(other.coefficients.len());
        let mut c = vec![0i64; n];
        for (i, &v) in self.coefficients.iter().enumerate() { c[i] += v; }
        for (i, &v) in other.coefficients.iter().enumerate() { c[i] -= v; }
        RingElement { coefficients: c }
    }

    pub fn scalar_mul(&self, s: i64) -> Self {
        RingElement { coefficients: self.coefficients.iter().map(|&c| c * s).collect() }
    }

    pub fn dimension(&self) -> i64 { self.coefficients.iter().sum() }

    pub fn is_zero(&self) -> bool { self.coefficients.iter().all(|&c| c == 0) }
}

impl fmt::Display for RingElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let parts: Vec<String> = self.coefficients.iter().enumerate()
            .filter(|(_, &c)| c != 0)
            .map(|(i, &c)| {
                if c == 1 { format!("ρ_{}", i) }
                else if c == -1 { format!("-ρ_{}", i) }
                else { format!("{}·ρ_{}", c, i) }
            })
            .collect();
        if parts.is_empty() { write!(f, "0") }
        else { write!(f, "{}", parts.join(" + ")) }
    }
}

#[derive(Debug, Clone)]
pub struct RepresentationRing {
    pub num_irreducibles: usize,
    pub tensor_table: Vec<Vec<RingElement>>,
}

impl RepresentationRing {
    pub fn new(num_irreducibles: usize) -> Self {
        let tensor_table = vec![vec![RingElement::zero(num_irreducibles); num_irreducibles]; num_irreducibles];
        RepresentationRing { num_irreducibles, tensor_table }
    }

    pub fn set_tensor_product(&mut self, i: usize, j: usize, result: RingElement) {
        self.tensor_table[i][j] = result;
    }

    pub fn tensor_product(&self, i: usize, j: usize) -> &RingElement {
        &self.tensor_table[i][j]
    }

    pub fn ring_multiply(&self, a: &RingElement, b: &RingElement) -> RingElement {
        let mut result = RingElement::zero(self.num_irreducibles);
        for (i, &ai) in a.coefficients.iter().enumerate() {
            if ai == 0 { continue; }
            for (j, &bj) in b.coefficients.iter().enumerate() {
                if bj == 0 { continue; }
                let prod = self.tensor_table[i][j].scalar_mul(ai * bj);
                result = result.add(&prod);
            }
        }
        result
    }
}

// =========================================================================
// Utility Functions
// =========================================================================

pub fn permutation_to_matrix(perm: &[usize]) -> Matrix {
    let n = perm.len();
    let mut m = Matrix::new(n, n);
    for (i, &j) in perm.iter().enumerate() { m.data[i][j] = 1.0; }
    m
}

pub fn random_orthogonal(n: usize, seed: u64) -> Matrix {
    // Simple pseudo-random orthogonal matrix via Gram-Schmidt
    let mut rng = seed;
    let mut a = Matrix::new(n, n);
    for i in 0..n {
        for j in 0..n {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            a.data[i][j] = ((rng >> 33) as f64) / (1u64 << 31) as f64 - 1.0;
        }
    }
    // Gram-Schmidt
    for i in 0..n {
        for j in 0..i {
            let mut dot = 0.0;
            for k in 0..n { dot += a.data[i][k] * a.data[j][k]; }
            for k in 0..n { a.data[i][k] -= dot * a.data[j][k]; }
        }
        let mut norm = 0.0;
        for k in 0..n { norm += a.data[i][k] * a.data[i][k]; }
        let norm = norm.sqrt();
        if norm > 1e-12 { for k in 0..n { a.data[i][k] /= norm; } }
    }
    a
}

pub fn representation_statistics(rep: &Representation) -> String {
    let mut out = String::new();
    out.push_str(&format!("Representation Statistics:\n"));
    out.push_str(&format!("  Dimension: {}\n", rep.dimension));
    out.push_str(&format!("  Group order: {}\n", rep.group_order));
    out.push_str(&format!("  Faithful: {}\n", rep.is_faithful()));
    out.push_str(&format!("  Kernel size: {}\n", rep.kernel().len()));
    out
}

pub fn print_character_table(table: &CharacterTable) -> String {
    format!("{}", table)
}

// =========================================================================
// Tests
// =========================================================================

// ===== Extended Matrix Repr Operations =====

#[derive(Debug, Clone)]
pub struct SchurDecomposition {
    pub unitary: Vec<Vec<f64>>,
    pub triangular: Vec<Vec<f64>>,
    pub size: usize,
}

impl SchurDecomposition {
    pub fn new(unitary: Vec<Vec<f64>>, triangular: Vec<Vec<f64>>, size: usize) -> Self {
        SchurDecomposition { unitary, triangular, size }
    }

    pub fn unitary_len(&self) -> usize {
        self.unitary.len()
    }

    pub fn unitary_is_empty(&self) -> bool {
        self.unitary.is_empty()
    }

    pub fn triangular_len(&self) -> usize {
        self.triangular.len()
    }

    pub fn triangular_is_empty(&self) -> bool {
        self.triangular.is_empty()
    }

    pub fn get_size(&self) -> usize {
        self.size
    }

    pub fn with_size(mut self, v: usize) -> Self {
        self.size = v; self
    }

}

impl fmt::Display for SchurDecomposition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SchurDecomposition({:?})", self.unitary)
    }
}

#[derive(Debug, Clone)]
pub struct SchurDecompositionBuilder {
    unitary: Vec<Vec<f64>>,
    triangular: Vec<Vec<f64>>,
    size: usize,
}

impl SchurDecompositionBuilder {
    pub fn new() -> Self {
        SchurDecompositionBuilder {
            unitary: Vec::new(),
            triangular: Vec::new(),
            size: 0,
        }
    }

    pub fn unitary(mut self, v: Vec<Vec<f64>>) -> Self { self.unitary = v; self }
    pub fn triangular(mut self, v: Vec<Vec<f64>>) -> Self { self.triangular = v; self }
    pub fn size(mut self, v: usize) -> Self { self.size = v; self }
}

#[derive(Debug, Clone)]
pub struct QrFactorization {
    pub q_matrix: Vec<Vec<f64>>,
    pub r_matrix: Vec<Vec<f64>>,
    pub size: usize,
}

impl QrFactorization {
    pub fn new(q_matrix: Vec<Vec<f64>>, r_matrix: Vec<Vec<f64>>, size: usize) -> Self {
        QrFactorization { q_matrix, r_matrix, size }
    }

    pub fn q_matrix_len(&self) -> usize {
        self.q_matrix.len()
    }

    pub fn q_matrix_is_empty(&self) -> bool {
        self.q_matrix.is_empty()
    }

    pub fn r_matrix_len(&self) -> usize {
        self.r_matrix.len()
    }

    pub fn r_matrix_is_empty(&self) -> bool {
        self.r_matrix.is_empty()
    }

    pub fn get_size(&self) -> usize {
        self.size
    }

    pub fn with_size(mut self, v: usize) -> Self {
        self.size = v; self
    }

}

impl fmt::Display for QrFactorization {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "QrFactorization({:?})", self.q_matrix)
    }
}

#[derive(Debug, Clone)]
pub struct QrFactorizationBuilder {
    q_matrix: Vec<Vec<f64>>,
    r_matrix: Vec<Vec<f64>>,
    size: usize,
}

impl QrFactorizationBuilder {
    pub fn new() -> Self {
        QrFactorizationBuilder {
            q_matrix: Vec::new(),
            r_matrix: Vec::new(),
            size: 0,
        }
    }

    pub fn q_matrix(mut self, v: Vec<Vec<f64>>) -> Self { self.q_matrix = v; self }
    pub fn r_matrix(mut self, v: Vec<Vec<f64>>) -> Self { self.r_matrix = v; self }
    pub fn size(mut self, v: usize) -> Self { self.size = v; self }
}

#[derive(Debug, Clone)]
pub struct SvdResult {
    pub u_matrix: Vec<Vec<f64>>,
    pub singular_values: Vec<f64>,
    pub v_matrix: Vec<Vec<f64>>,
    pub size: usize,
}

impl SvdResult {
    pub fn new(u_matrix: Vec<Vec<f64>>, singular_values: Vec<f64>, v_matrix: Vec<Vec<f64>>, size: usize) -> Self {
        SvdResult { u_matrix, singular_values, v_matrix, size }
    }

    pub fn u_matrix_len(&self) -> usize {
        self.u_matrix.len()
    }

    pub fn u_matrix_is_empty(&self) -> bool {
        self.u_matrix.is_empty()
    }

    pub fn singular_values_len(&self) -> usize {
        self.singular_values.len()
    }

    pub fn singular_values_is_empty(&self) -> bool {
        self.singular_values.is_empty()
    }

    pub fn v_matrix_len(&self) -> usize {
        self.v_matrix.len()
    }

    pub fn v_matrix_is_empty(&self) -> bool {
        self.v_matrix.is_empty()
    }

    pub fn get_size(&self) -> usize {
        self.size
    }

    pub fn with_size(mut self, v: usize) -> Self {
        self.size = v; self
    }

}

impl fmt::Display for SvdResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SvdResult({:?})", self.u_matrix)
    }
}

#[derive(Debug, Clone)]
pub struct SvdResultBuilder {
    u_matrix: Vec<Vec<f64>>,
    singular_values: Vec<f64>,
    v_matrix: Vec<Vec<f64>>,
    size: usize,
}

impl SvdResultBuilder {
    pub fn new() -> Self {
        SvdResultBuilder {
            u_matrix: Vec::new(),
            singular_values: Vec::new(),
            v_matrix: Vec::new(),
            size: 0,
        }
    }

    pub fn u_matrix(mut self, v: Vec<Vec<f64>>) -> Self { self.u_matrix = v; self }
    pub fn singular_values(mut self, v: Vec<f64>) -> Self { self.singular_values = v; self }
    pub fn v_matrix(mut self, v: Vec<Vec<f64>>) -> Self { self.v_matrix = v; self }
    pub fn size(mut self, v: usize) -> Self { self.size = v; self }
}

#[derive(Debug, Clone)]
pub struct MatrixExponential {
    pub input: Vec<Vec<f64>>,
    pub result: Vec<Vec<f64>>,
    pub terms_used: usize,
}

impl MatrixExponential {
    pub fn new(input: Vec<Vec<f64>>, result: Vec<Vec<f64>>, terms_used: usize) -> Self {
        MatrixExponential { input, result, terms_used }
    }

    pub fn input_len(&self) -> usize {
        self.input.len()
    }

    pub fn input_is_empty(&self) -> bool {
        self.input.is_empty()
    }

    pub fn result_len(&self) -> usize {
        self.result.len()
    }

    pub fn result_is_empty(&self) -> bool {
        self.result.is_empty()
    }

    pub fn get_terms_used(&self) -> usize {
        self.terms_used
    }

    pub fn with_terms_used(mut self, v: usize) -> Self {
        self.terms_used = v; self
    }

}

impl fmt::Display for MatrixExponential {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MatrixExponential({:?})", self.input)
    }
}

#[derive(Debug, Clone)]
pub struct MatrixExponentialBuilder {
    input: Vec<Vec<f64>>,
    result: Vec<Vec<f64>>,
    terms_used: usize,
}

impl MatrixExponentialBuilder {
    pub fn new() -> Self {
        MatrixExponentialBuilder {
            input: Vec::new(),
            result: Vec::new(),
            terms_used: 0,
        }
    }

    pub fn input(mut self, v: Vec<Vec<f64>>) -> Self { self.input = v; self }
    pub fn result(mut self, v: Vec<Vec<f64>>) -> Self { self.result = v; self }
    pub fn terms_used(mut self, v: usize) -> Self { self.terms_used = v; self }
}

#[derive(Debug, Clone)]
pub struct MatrixLogarithm {
    pub input: Vec<Vec<f64>>,
    pub result: Vec<Vec<f64>>,
    pub converged: bool,
}

impl MatrixLogarithm {
    pub fn new(input: Vec<Vec<f64>>, result: Vec<Vec<f64>>, converged: bool) -> Self {
        MatrixLogarithm { input, result, converged }
    }

    pub fn input_len(&self) -> usize {
        self.input.len()
    }

    pub fn input_is_empty(&self) -> bool {
        self.input.is_empty()
    }

    pub fn result_len(&self) -> usize {
        self.result.len()
    }

    pub fn result_is_empty(&self) -> bool {
        self.result.is_empty()
    }

    pub fn get_converged(&self) -> bool {
        self.converged
    }

    pub fn with_converged(mut self, v: bool) -> Self {
        self.converged = v; self
    }

}

impl fmt::Display for MatrixLogarithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MatrixLogarithm({:?})", self.input)
    }
}

#[derive(Debug, Clone)]
pub struct MatrixLogarithmBuilder {
    input: Vec<Vec<f64>>,
    result: Vec<Vec<f64>>,
    converged: bool,
}

impl MatrixLogarithmBuilder {
    pub fn new() -> Self {
        MatrixLogarithmBuilder {
            input: Vec::new(),
            result: Vec::new(),
            converged: false,
        }
    }

    pub fn input(mut self, v: Vec<Vec<f64>>) -> Self { self.input = v; self }
    pub fn result(mut self, v: Vec<Vec<f64>>) -> Self { self.result = v; self }
    pub fn converged(mut self, v: bool) -> Self { self.converged = v; self }
}

#[derive(Debug, Clone)]
pub struct CayleyHamilton {
    pub coefficients: Vec<f64>,
    pub matrix_size: usize,
    pub verified: bool,
}

impl CayleyHamilton {
    pub fn new(coefficients: Vec<f64>, matrix_size: usize, verified: bool) -> Self {
        CayleyHamilton { coefficients, matrix_size, verified }
    }

    pub fn coefficients_len(&self) -> usize {
        self.coefficients.len()
    }

    pub fn coefficients_is_empty(&self) -> bool {
        self.coefficients.is_empty()
    }

    pub fn get_matrix_size(&self) -> usize {
        self.matrix_size
    }

    pub fn get_verified(&self) -> bool {
        self.verified
    }

    pub fn with_matrix_size(mut self, v: usize) -> Self {
        self.matrix_size = v; self
    }

    pub fn with_verified(mut self, v: bool) -> Self {
        self.verified = v; self
    }

}

impl fmt::Display for CayleyHamilton {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CayleyHamilton({:?})", self.coefficients)
    }
}

#[derive(Debug, Clone)]
pub struct CayleyHamiltonBuilder {
    coefficients: Vec<f64>,
    matrix_size: usize,
    verified: bool,
}

impl CayleyHamiltonBuilder {
    pub fn new() -> Self {
        CayleyHamiltonBuilder {
            coefficients: Vec::new(),
            matrix_size: 0,
            verified: false,
        }
    }

    pub fn coefficients(mut self, v: Vec<f64>) -> Self { self.coefficients = v; self }
    pub fn matrix_size(mut self, v: usize) -> Self { self.matrix_size = v; self }
    pub fn verified(mut self, v: bool) -> Self { self.verified = v; self }
}

#[derive(Debug, Clone)]
pub struct MinimalPolynomial {
    pub coefficients: Vec<f64>,
    pub degree: usize,
}

impl MinimalPolynomial {
    pub fn new(coefficients: Vec<f64>, degree: usize) -> Self {
        MinimalPolynomial { coefficients, degree }
    }

    pub fn coefficients_len(&self) -> usize {
        self.coefficients.len()
    }

    pub fn coefficients_is_empty(&self) -> bool {
        self.coefficients.is_empty()
    }

    pub fn get_degree(&self) -> usize {
        self.degree
    }

    pub fn with_degree(mut self, v: usize) -> Self {
        self.degree = v; self
    }

}

impl fmt::Display for MinimalPolynomial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MinimalPolynomial({:?})", self.coefficients)
    }
}

#[derive(Debug, Clone)]
pub struct MinimalPolynomialBuilder {
    coefficients: Vec<f64>,
    degree: usize,
}

impl MinimalPolynomialBuilder {
    pub fn new() -> Self {
        MinimalPolynomialBuilder {
            coefficients: Vec::new(),
            degree: 0,
        }
    }

    pub fn coefficients(mut self, v: Vec<f64>) -> Self { self.coefficients = v; self }
    pub fn degree(mut self, v: usize) -> Self { self.degree = v; self }
}

#[derive(Debug, Clone)]
pub struct JordanBlock {
    pub eigenvalue: f64,
    pub size: usize,
    pub block_index: usize,
}

impl JordanBlock {
    pub fn new(eigenvalue: f64, size: usize, block_index: usize) -> Self {
        JordanBlock { eigenvalue, size, block_index }
    }

    pub fn get_eigenvalue(&self) -> f64 {
        self.eigenvalue
    }

    pub fn get_size(&self) -> usize {
        self.size
    }

    pub fn get_block_index(&self) -> usize {
        self.block_index
    }

    pub fn with_eigenvalue(mut self, v: f64) -> Self {
        self.eigenvalue = v; self
    }

    pub fn with_size(mut self, v: usize) -> Self {
        self.size = v; self
    }

    pub fn with_block_index(mut self, v: usize) -> Self {
        self.block_index = v; self
    }

}

impl fmt::Display for JordanBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "JordanBlock({:?})", self.eigenvalue)
    }
}

#[derive(Debug, Clone)]
pub struct JordanBlockBuilder {
    eigenvalue: f64,
    size: usize,
    block_index: usize,
}

impl JordanBlockBuilder {
    pub fn new() -> Self {
        JordanBlockBuilder {
            eigenvalue: 0.0,
            size: 0,
            block_index: 0,
        }
    }

    pub fn eigenvalue(mut self, v: f64) -> Self { self.eigenvalue = v; self }
    pub fn size(mut self, v: usize) -> Self { self.size = v; self }
    pub fn block_index(mut self, v: usize) -> Self { self.block_index = v; self }
}

#[derive(Debug, Clone)]
pub struct JordanForm {
    pub blocks: Vec<(f64, usize)>,
    pub transform: Vec<Vec<f64>>,
    pub size: usize,
}

impl JordanForm {
    pub fn new(blocks: Vec<(f64, usize)>, transform: Vec<Vec<f64>>, size: usize) -> Self {
        JordanForm { blocks, transform, size }
    }

    pub fn blocks_len(&self) -> usize {
        self.blocks.len()
    }

    pub fn blocks_is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    pub fn transform_len(&self) -> usize {
        self.transform.len()
    }

    pub fn transform_is_empty(&self) -> bool {
        self.transform.is_empty()
    }

    pub fn get_size(&self) -> usize {
        self.size
    }

    pub fn with_size(mut self, v: usize) -> Self {
        self.size = v; self
    }

}

impl fmt::Display for JordanForm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "JordanForm({:?})", self.blocks)
    }
}

#[derive(Debug, Clone)]
pub struct JordanFormBuilder {
    blocks: Vec<(f64, usize)>,
    transform: Vec<Vec<f64>>,
    size: usize,
}

impl JordanFormBuilder {
    pub fn new() -> Self {
        JordanFormBuilder {
            blocks: Vec::new(),
            transform: Vec::new(),
            size: 0,
        }
    }

    pub fn blocks(mut self, v: Vec<(f64, usize)>) -> Self { self.blocks = v; self }
    pub fn transform(mut self, v: Vec<Vec<f64>>) -> Self { self.transform = v; self }
    pub fn size(mut self, v: usize) -> Self { self.size = v; self }
}

#[derive(Debug, Clone)]
pub struct OrthogonalProjection {
    pub subspace_basis: Vec<Vec<f64>>,
    pub projection_matrix: Vec<Vec<f64>>,
    pub rank: usize,
}

impl OrthogonalProjection {
    pub fn new(subspace_basis: Vec<Vec<f64>>, projection_matrix: Vec<Vec<f64>>, rank: usize) -> Self {
        OrthogonalProjection { subspace_basis, projection_matrix, rank }
    }

    pub fn subspace_basis_len(&self) -> usize {
        self.subspace_basis.len()
    }

    pub fn subspace_basis_is_empty(&self) -> bool {
        self.subspace_basis.is_empty()
    }

    pub fn projection_matrix_len(&self) -> usize {
        self.projection_matrix.len()
    }

    pub fn projection_matrix_is_empty(&self) -> bool {
        self.projection_matrix.is_empty()
    }

    pub fn get_rank(&self) -> usize {
        self.rank
    }

    pub fn with_rank(mut self, v: usize) -> Self {
        self.rank = v; self
    }

}

impl fmt::Display for OrthogonalProjection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OrthogonalProjection({:?})", self.subspace_basis)
    }
}

#[derive(Debug, Clone)]
pub struct OrthogonalProjectionBuilder {
    subspace_basis: Vec<Vec<f64>>,
    projection_matrix: Vec<Vec<f64>>,
    rank: usize,
}

impl OrthogonalProjectionBuilder {
    pub fn new() -> Self {
        OrthogonalProjectionBuilder {
            subspace_basis: Vec::new(),
            projection_matrix: Vec::new(),
            rank: 0,
        }
    }

    pub fn subspace_basis(mut self, v: Vec<Vec<f64>>) -> Self { self.subspace_basis = v; self }
    pub fn projection_matrix(mut self, v: Vec<Vec<f64>>) -> Self { self.projection_matrix = v; self }
    pub fn rank(mut self, v: usize) -> Self { self.rank = v; self }
}

#[derive(Debug, Clone)]
pub struct GramSchmidt {
    pub input_vectors: Vec<Vec<f64>>,
    pub orthogonal_vectors: Vec<Vec<f64>>,
    pub dimension: usize,
}

impl GramSchmidt {
    pub fn new(input_vectors: Vec<Vec<f64>>, orthogonal_vectors: Vec<Vec<f64>>, dimension: usize) -> Self {
        GramSchmidt { input_vectors, orthogonal_vectors, dimension }
    }

    pub fn input_vectors_len(&self) -> usize {
        self.input_vectors.len()
    }

    pub fn input_vectors_is_empty(&self) -> bool {
        self.input_vectors.is_empty()
    }

    pub fn orthogonal_vectors_len(&self) -> usize {
        self.orthogonal_vectors.len()
    }

    pub fn orthogonal_vectors_is_empty(&self) -> bool {
        self.orthogonal_vectors.is_empty()
    }

    pub fn get_dimension(&self) -> usize {
        self.dimension
    }

    pub fn with_dimension(mut self, v: usize) -> Self {
        self.dimension = v; self
    }

}

impl fmt::Display for GramSchmidt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GramSchmidt({:?})", self.input_vectors)
    }
}

#[derive(Debug, Clone)]
pub struct GramSchmidtBuilder {
    input_vectors: Vec<Vec<f64>>,
    orthogonal_vectors: Vec<Vec<f64>>,
    dimension: usize,
}

impl GramSchmidtBuilder {
    pub fn new() -> Self {
        GramSchmidtBuilder {
            input_vectors: Vec::new(),
            orthogonal_vectors: Vec::new(),
            dimension: 0,
        }
    }

    pub fn input_vectors(mut self, v: Vec<Vec<f64>>) -> Self { self.input_vectors = v; self }
    pub fn orthogonal_vectors(mut self, v: Vec<Vec<f64>>) -> Self { self.orthogonal_vectors = v; self }
    pub fn dimension(mut self, v: usize) -> Self { self.dimension = v; self }
}

#[derive(Debug, Clone)]
pub struct PowerIteration {
    pub eigenvalue: f64,
    pub eigenvector: Vec<f64>,
    pub iterations: usize,
    pub converged: bool,
}

impl PowerIteration {
    pub fn new(eigenvalue: f64, eigenvector: Vec<f64>, iterations: usize, converged: bool) -> Self {
        PowerIteration { eigenvalue, eigenvector, iterations, converged }
    }

    pub fn get_eigenvalue(&self) -> f64 {
        self.eigenvalue
    }

    pub fn eigenvector_len(&self) -> usize {
        self.eigenvector.len()
    }

    pub fn eigenvector_is_empty(&self) -> bool {
        self.eigenvector.is_empty()
    }

    pub fn get_iterations(&self) -> usize {
        self.iterations
    }

    pub fn get_converged(&self) -> bool {
        self.converged
    }

    pub fn with_eigenvalue(mut self, v: f64) -> Self {
        self.eigenvalue = v; self
    }

    pub fn with_iterations(mut self, v: usize) -> Self {
        self.iterations = v; self
    }

    pub fn with_converged(mut self, v: bool) -> Self {
        self.converged = v; self
    }

}

impl fmt::Display for PowerIteration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PowerIteration({:?})", self.eigenvalue)
    }
}

#[derive(Debug, Clone)]
pub struct PowerIterationBuilder {
    eigenvalue: f64,
    eigenvector: Vec<f64>,
    iterations: usize,
    converged: bool,
}

impl PowerIterationBuilder {
    pub fn new() -> Self {
        PowerIterationBuilder {
            eigenvalue: 0.0,
            eigenvector: Vec::new(),
            iterations: 0,
            converged: false,
        }
    }

    pub fn eigenvalue(mut self, v: f64) -> Self { self.eigenvalue = v; self }
    pub fn eigenvector(mut self, v: Vec<f64>) -> Self { self.eigenvector = v; self }
    pub fn iterations(mut self, v: usize) -> Self { self.iterations = v; self }
    pub fn converged(mut self, v: bool) -> Self { self.converged = v; self }
}

#[derive(Debug, Clone)]
pub struct MatrixAnalysis {
    pub data: Vec<Vec<f64>>,
    pub size: usize,
    pub computed: bool,
    pub label: String,
    pub threshold: f64,
}

impl MatrixAnalysis {
    pub fn new(size: usize) -> Self {
        let data = vec![vec![0.0; size]; size];
        MatrixAnalysis { data, size, computed: false, label: "Matrix".to_string(), threshold: 0.01 }
    }

    pub fn with_threshold(mut self, t: f64) -> Self {
        self.threshold = t; self
    }

    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        if i < self.size && j < self.size { self.data[i][j] = v; }
    }

    pub fn get(&self, i: usize, j: usize) -> f64 {
        if i < self.size && j < self.size { self.data[i][j] } else { 0.0 }
    }

    pub fn row_sum(&self, i: usize) -> f64 {
        if i < self.size { self.data[i].iter().sum() } else { 0.0 }
    }

    pub fn col_sum(&self, j: usize) -> f64 {
        if j < self.size { (0..self.size).map(|i| self.data[i][j]).sum() } else { 0.0 }
    }

    pub fn total_sum(&self) -> f64 {
        self.data.iter().flat_map(|r| r.iter()).sum()
    }

    pub fn max_value(&self) -> f64 {
        self.data.iter().flat_map(|r| r.iter()).cloned().fold(f64::NEG_INFINITY, f64::max)
    }

    pub fn min_value(&self) -> f64 {
        self.data.iter().flat_map(|r| r.iter()).cloned().fold(f64::INFINITY, f64::min)
    }

    pub fn above_threshold(&self) -> Vec<(usize, usize, f64)> {
        let mut result = Vec::new();
        for i in 0..self.size {
            for j in 0..self.size {
                if self.data[i][j] > self.threshold {
                    result.push((i, j, self.data[i][j]));
                }
            }
        }
        result
    }

    pub fn normalize(&mut self) {
        let total = self.total_sum();
        if total > 0.0 {
            for i in 0..self.size {
                for j in 0..self.size {
                    self.data[i][j] /= total;
                }
            }
        }
        self.computed = true;
    }

    pub fn transpose(&self) -> Self {
        let mut result = Self::new(self.size);
        for i in 0..self.size {
            for j in 0..self.size {
                result.data[i][j] = self.data[j][i];
            }
        }
        result
    }

    pub fn multiply(&self, other: &Self) -> Self {
        assert_eq!(self.size, other.size);
        let mut result = Self::new(self.size);
        for i in 0..self.size {
            for j in 0..self.size {
                let mut sum = 0.0;
                for k in 0..self.size {
                    sum += self.data[i][k] * other.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        result
    }

    pub fn frobenius_norm(&self) -> f64 {
        self.data.iter().flat_map(|r| r.iter()).map(|&v| v * v).sum::<f64>().sqrt()
    }

    pub fn trace(&self) -> f64 {
        (0..self.size).map(|i| self.data[i][i]).sum()
    }

    pub fn diagonal(&self) -> Vec<f64> {
        (0..self.size).map(|i| self.data[i][i]).collect()
    }

    pub fn is_symmetric(&self) -> bool {
        for i in 0..self.size {
            for j in 0..self.size {
                if (self.data[i][j] - self.data[j][i]).abs() > 1e-10 { return false; }
            }
        }
        true
    }

}

impl fmt::Display for MatrixAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MatrixAnalysis({:?})", self.data)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MatrixStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Skipped,
}

impl fmt::Display for MatrixStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatrixStatus::Pending => write!(f, "pending"),
            MatrixStatus::InProgress => write!(f, "inprogress"),
            MatrixStatus::Completed => write!(f, "completed"),
            MatrixStatus::Failed => write!(f, "failed"),
            MatrixStatus::Skipped => write!(f, "skipped"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MatrixPriority {
    Critical,
    High,
    Medium,
    Low,
    None,
}

impl fmt::Display for MatrixPriority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatrixPriority::Critical => write!(f, "critical"),
            MatrixPriority::High => write!(f, "high"),
            MatrixPriority::Medium => write!(f, "medium"),
            MatrixPriority::Low => write!(f, "low"),
            MatrixPriority::None => write!(f, "none"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MatrixMode {
    Strict,
    Relaxed,
    Permissive,
    Custom,
}

impl fmt::Display for MatrixMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatrixMode::Strict => write!(f, "strict"),
            MatrixMode::Relaxed => write!(f, "relaxed"),
            MatrixMode::Permissive => write!(f, "permissive"),
            MatrixMode::Custom => write!(f, "custom"),
        }
    }
}

pub fn matrix_mean(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    data.iter().sum::<f64>() / data.len() as f64
}

pub fn matrix_variance(data: &[f64]) -> f64 {
    if data.len() < 2 { return 0.0; }
    let mean = matrix_mean(data);
    data.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / (data.len() - 1) as f64
}

pub fn matrix_std_dev(data: &[f64]) -> f64 {
    matrix_variance(data).sqrt()
}

pub fn matrix_median(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    if n % 2 == 0 { (sorted[n/2 - 1] + sorted[n/2]) / 2.0 } else { sorted[n/2] }
}

pub fn matrix_percentile(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((sorted.len() as f64 - 1.0) * 0.95).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

pub fn matrix_entropy(data: &[f64]) -> f64 {
    let total: f64 = data.iter().sum();
    if total <= 0.0 { return 0.0; }
    let mut h = 0.0;
    for &x in data {
        if x > 0.0 {
            let p = x / total;
            h -= p * p.ln();
        }
    }
    h
}

pub fn matrix_gini(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    let n = data.len();
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let sum: f64 = sorted.iter().sum();
    if sum == 0.0 { return 0.0; }
    let mut g = 0.0;
    for (i, &x) in sorted.iter().enumerate() {
        g += (2.0 * (i + 1) as f64 - n as f64 - 1.0) * x;
    }
    g / (n as f64 * sum)
}

pub fn matrix_covariance(data: &[f64]) -> f64 {
    if data.len() < 4 { return 0.0; }
    let n = data.len() / 2;
    let x: Vec<f64> = data[..n].to_vec();
    let y: Vec<f64> = data[n..2*n].to_vec();
    let mx = matrix_mean(&x);
    let my = matrix_mean(&y);
    x.iter().zip(y.iter()).map(|(&xi, &yi)| (xi - mx) * (yi - my)).sum::<f64>() / (n - 1) as f64
}

pub fn matrix_correlation(data: &[f64]) -> f64 {
    if data.len() < 4 { return 0.0; }
    let n = data.len() / 2;
    let cov = matrix_covariance(data);
    let sx = matrix_std_dev(&data[..n]);
    let sy = matrix_std_dev(&data[n..2*n]);
    if sx * sy == 0.0 { 0.0 } else { cov / (sx * sy) }
}

pub fn matrix_kurtosis(data: &[f64]) -> f64 {
    if data.len() < 4 { return 0.0; }
    let m = matrix_mean(data);
    let s = matrix_std_dev(data);
    if s == 0.0 { return 0.0; }
    let n = data.len() as f64;
    let k = data.iter().map(|&x| ((x - m) / s).powi(4)).sum::<f64>() / n;
    k - 3.0
}

pub fn matrix_skewness(data: &[f64]) -> f64 {
    if data.len() < 3 { return 0.0; }
    let m = matrix_mean(data);
    let s = matrix_std_dev(data);
    if s == 0.0 { return 0.0; }
    let n = data.len() as f64;
    data.iter().map(|&x| ((x - m) / s).powi(3)).sum::<f64>() / n
}

pub fn matrix_harmonic_mean(data: &[f64]) -> f64 {
    if data.is_empty() || data.iter().any(|&x| x <= 0.0) { return 0.0; }
    let n = data.len() as f64;
    n / data.iter().map(|&x| 1.0 / x).sum::<f64>()
}

pub fn matrix_geometric_mean(data: &[f64]) -> f64 {
    if data.is_empty() || data.iter().any(|&x| x <= 0.0) { return 0.0; }
    let n = data.len() as f64;
    (data.iter().map(|&x| x.ln()).sum::<f64>() / n).exp()
}

/// Iterator over matrix analysis results.
#[derive(Debug, Clone)]
pub struct MatrixResultIterator {
    items: Vec<(usize, f64)>,
    position: usize,
}

impl MatrixResultIterator {
    pub fn new(items: Vec<(usize, f64)>) -> Self {
        MatrixResultIterator { items, position: 0 }
    }
    pub fn remaining(&self) -> usize { self.items.len() - self.position }
}

impl Iterator for MatrixResultIterator {
    type Item = (usize, f64);
    fn next(&mut self) -> Option<Self::Item> {
        if self.position < self.items.len() {
            let item = self.items[self.position];
            self.position += 1;
            Some(item)
        } else { None }
    }
}

/// Convert SchurDecomposition description to a summary string.
pub fn schurdecomposition_to_summary(item: &SchurDecomposition) -> String {
    format!("SchurDecomposition: {:?}", item)
}

/// Convert QrFactorization description to a summary string.
pub fn qrfactorization_to_summary(item: &QrFactorization) -> String {
    format!("QrFactorization: {:?}", item)
}

/// Convert SvdResult description to a summary string.
pub fn svdresult_to_summary(item: &SvdResult) -> String {
    format!("SvdResult: {:?}", item)
}

/// Convert MatrixExponential description to a summary string.
pub fn matrixexponential_to_summary(item: &MatrixExponential) -> String {
    format!("MatrixExponential: {:?}", item)
}

/// Convert MatrixLogarithm description to a summary string.
pub fn matrixlogarithm_to_summary(item: &MatrixLogarithm) -> String {
    format!("MatrixLogarithm: {:?}", item)
}

/// Convert CayleyHamilton description to a summary string.
pub fn cayleyhamilton_to_summary(item: &CayleyHamilton) -> String {
    format!("CayleyHamilton: {:?}", item)
}

/// Convert MinimalPolynomial description to a summary string.
pub fn minimalpolynomial_to_summary(item: &MinimalPolynomial) -> String {
    format!("MinimalPolynomial: {:?}", item)
}

/// Convert JordanBlock description to a summary string.
pub fn jordanblock_to_summary(item: &JordanBlock) -> String {
    format!("JordanBlock: {:?}", item)
}

/// Convert JordanForm description to a summary string.
pub fn jordanform_to_summary(item: &JordanForm) -> String {
    format!("JordanForm: {:?}", item)
}

/// Convert OrthogonalProjection description to a summary string.
pub fn orthogonalprojection_to_summary(item: &OrthogonalProjection) -> String {
    format!("OrthogonalProjection: {:?}", item)
}

/// Convert GramSchmidt description to a summary string.
pub fn gramschmidt_to_summary(item: &GramSchmidt) -> String {
    format!("GramSchmidt: {:?}", item)
}

/// Batch processor for matrix operations.
#[derive(Debug, Clone)]
pub struct MatrixBatchProcessor {
    pub batch_size: usize,
    pub processed: usize,
    pub errors: Vec<String>,
    pub results: Vec<f64>,
}

impl MatrixBatchProcessor {
    pub fn new(batch_size: usize) -> Self {
        MatrixBatchProcessor { batch_size, processed: 0, errors: Vec::new(), results: Vec::new() }
    }
    pub fn process_batch(&mut self, data: &[f64]) {
        for chunk in data.chunks(self.batch_size) {
            let sum: f64 = chunk.iter().sum();
            self.results.push(sum / chunk.len() as f64);
            self.processed += chunk.len();
        }
    }
    pub fn success_rate(&self) -> f64 {
        if self.processed == 0 { return 0.0; }
        1.0 - (self.errors.len() as f64 / self.processed as f64)
    }
    pub fn average_result(&self) -> f64 {
        if self.results.is_empty() { return 0.0; }
        self.results.iter().sum::<f64>() / self.results.len() as f64
    }
    pub fn reset(&mut self) { self.processed = 0; self.errors.clear(); self.results.clear(); }
}

impl fmt::Display for MatrixBatchProcessor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MatrixBatch(processed={}, errors={})", self.processed, self.errors.len())
    }
}

/// Detailed report for matrix analysis.
#[derive(Debug, Clone)]
pub struct MatrixReport {
    pub title: String,
    pub sections: Vec<(String, Vec<String>)>,
    pub metrics: Vec<(String, f64)>,
    pub warnings: Vec<String>,
    pub timestamp: u64,
}

impl MatrixReport {
    pub fn new(title: impl Into<String>) -> Self {
        MatrixReport { title: title.into(), sections: Vec::new(), metrics: Vec::new(), warnings: Vec::new(), timestamp: 0 }
    }
    pub fn add_section(&mut self, name: impl Into<String>, content: Vec<String>) {
        self.sections.push((name.into(), content));
    }
    pub fn add_metric(&mut self, name: impl Into<String>, value: f64) {
        self.metrics.push((name.into(), value));
    }
    pub fn add_warning(&mut self, warning: impl Into<String>) {
        self.warnings.push(warning.into());
    }
    pub fn total_metrics(&self) -> usize { self.metrics.len() }
    pub fn has_warnings(&self) -> bool { !self.warnings.is_empty() }
    pub fn metric_sum(&self) -> f64 { self.metrics.iter().map(|(_, v)| v).sum() }
    pub fn render_text(&self) -> String {
        let mut out = format!("=== {} ===\n", self.title);
        for (name, content) in &self.sections {
            out.push_str(&format!("\n--- {} ---\n", name));
            for line in content {
                out.push_str(&format!("  {}\n", line));
            }
        }
        out.push_str("\nMetrics:\n");
        for (name, val) in &self.metrics {
            out.push_str(&format!("  {}: {:.4}\n", name, val));
        }
        if !self.warnings.is_empty() {
            out.push_str("\nWarnings:\n");
            for w in &self.warnings {
                out.push_str(&format!("  ! {}\n", w));
            }
        }
        out
    }
}

impl fmt::Display for MatrixReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MatrixReport({})", self.title)
    }
}

/// Configuration for matrix analysis.
#[derive(Debug, Clone)]
pub struct MatrixConfig {
    pub verbose: bool,
    pub max_iterations: usize,
    pub tolerance: f64,
    pub timeout_ms: u64,
    pub parallel: bool,
    pub output_format: String,
}

impl MatrixConfig {
    pub fn default_config() -> Self {
        MatrixConfig {
            verbose: false, max_iterations: 1000, tolerance: 1e-6,
            timeout_ms: 30000, parallel: false, output_format: "text".to_string(),
        }
    }
    pub fn with_verbose(mut self, v: bool) -> Self { self.verbose = v; self }
    pub fn with_max_iterations(mut self, n: usize) -> Self { self.max_iterations = n; self }
    pub fn with_tolerance(mut self, t: f64) -> Self { self.tolerance = t; self }
    pub fn with_timeout(mut self, ms: u64) -> Self { self.timeout_ms = ms; self }
    pub fn with_parallel(mut self, p: bool) -> Self { self.parallel = p; self }
    pub fn with_output_format(mut self, fmt: impl Into<String>) -> Self { self.output_format = fmt.into(); self }
}

impl fmt::Display for MatrixConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MatrixConfig(iters={}, tol={:.0e})", self.max_iterations, self.tolerance)
    }
}

/// Histogram for matrix data distribution.
#[derive(Debug, Clone)]
pub struct MatrixHistogram {
    pub bins: Vec<usize>,
    pub bin_edges: Vec<f64>,
    pub total_count: usize,
}

impl MatrixHistogram {
    pub fn from_data(data: &[f64], num_bins: usize) -> Self {
        if data.is_empty() || num_bins == 0 {
            return MatrixHistogram { bins: Vec::new(), bin_edges: Vec::new(), total_count: 0 };
        }
        let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max_val - min_val;
        let bin_width = if range == 0.0 { 1.0 } else { range / num_bins as f64 };
        let mut bins = vec![0usize; num_bins];
        let mut bin_edges = Vec::with_capacity(num_bins + 1);
        for i in 0..=num_bins { bin_edges.push(min_val + i as f64 * bin_width); }
        for &val in data {
            let idx = ((val - min_val) / bin_width).floor() as usize;
            let idx = idx.min(num_bins - 1);
            bins[idx] += 1;
        }
        MatrixHistogram { bins, bin_edges, total_count: data.len() }
    }
    pub fn num_bins(&self) -> usize { self.bins.len() }
    pub fn max_bin(&self) -> usize { self.bins.iter().cloned().max().unwrap_or(0) }
    pub fn mean_bin(&self) -> f64 {
        if self.bins.is_empty() { return 0.0; }
        self.bins.iter().sum::<usize>() as f64 / self.bins.len() as f64
    }
    pub fn render_ascii(&self, width: usize) -> String {
        let max = self.max_bin();
        let mut out = String::new();
        for (i, &count) in self.bins.iter().enumerate() {
            let bar_len = if max == 0 { 0 } else { count * width / max };
            let bar: String = std::iter::repeat('#').take(bar_len).collect();
            out.push_str(&format!("[{:.2}-{:.2}] {} {}\n",
                self.bin_edges[i], self.bin_edges[i + 1], bar, count));
        }
        out
    }
}

impl fmt::Display for MatrixHistogram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Histogram(bins={}, total={})", self.num_bins(), self.total_count)
    }
}

/// Adjacency/weight matrix for matrix graph analysis.
#[derive(Debug, Clone)]
pub struct MatrixGraph {
    pub adjacency: Vec<Vec<bool>>,
    pub weights: Vec<Vec<f64>>,
    pub node_count: usize,
    pub edge_count: usize,
    pub node_labels: Vec<String>,
}

impl MatrixGraph {
    pub fn new(n: usize) -> Self {
        MatrixGraph {
            adjacency: vec![vec![false; n]; n],
            weights: vec![vec![0.0; n]; n],
            node_count: n, edge_count: 0,
            node_labels: (0..n).map(|i| format!("n{}", i)).collect(),
        }
    }
    pub fn add_edge(&mut self, from: usize, to: usize, weight: f64) {
        if from < self.node_count && to < self.node_count && !self.adjacency[from][to] {
            self.adjacency[from][to] = true;
            self.weights[from][to] = weight;
            self.edge_count += 1;
        }
    }
    pub fn remove_edge(&mut self, from: usize, to: usize) {
        if from < self.node_count && to < self.node_count && self.adjacency[from][to] {
            self.adjacency[from][to] = false;
            self.weights[from][to] = 0.0;
            self.edge_count -= 1;
        }
    }
    pub fn has_edge(&self, from: usize, to: usize) -> bool {
        from < self.node_count && to < self.node_count && self.adjacency[from][to]
    }
    pub fn weight(&self, from: usize, to: usize) -> f64 { self.weights[from][to] }
    pub fn out_degree(&self, node: usize) -> usize {
        (0..self.node_count).filter(|&j| self.adjacency[node][j]).count()
    }
    pub fn in_degree(&self, node: usize) -> usize {
        (0..self.node_count).filter(|&i| self.adjacency[i][node]).count()
    }
    pub fn neighbors(&self, node: usize) -> Vec<usize> {
        (0..self.node_count).filter(|&j| self.adjacency[node][j]).collect()
    }
    pub fn density(&self) -> f64 {
        if self.node_count <= 1 { return 0.0; }
        self.edge_count as f64 / (self.node_count * (self.node_count - 1)) as f64
    }
    pub fn is_acyclic(&self) -> bool {
        let n = self.node_count;
        let mut visited = vec![0u8; n];
        fn dfs_cycle_matrix(v: usize, adj: &[Vec<bool>], visited: &mut [u8]) -> bool {
            visited[v] = 1;
            for w in 0..adj.len() { if adj[v][w] {
                if visited[w] == 1 { return true; }
                if visited[w] == 0 && dfs_cycle_matrix(w, adj, visited) { return true; }
            }}
            visited[v] = 2; false
        }
        for i in 0..n {
            if visited[i] == 0 && dfs_cycle_matrix(i, &self.adjacency, &mut visited) { return false; }
        }
        true
    }
    pub fn topological_sort(&self) -> Option<Vec<usize>> {
        let n = self.node_count;
        let mut in_deg: Vec<usize> = (0..n).map(|j| self.in_degree(j)).collect();
        let mut queue: Vec<usize> = (0..n).filter(|&i| in_deg[i] == 0).collect();
        let mut result = Vec::new();
        while let Some(v) = queue.pop() {
            result.push(v);
            for j in 0..n { if self.adjacency[v][j] {
                in_deg[j] -= 1;
                if in_deg[j] == 0 { queue.push(j); }
            }}
        }
        if result.len() == n { Some(result) } else { None }
    }
    pub fn shortest_path_dijkstra(&self, start: usize) -> Vec<f64> {
        let n = self.node_count;
        let mut dist = vec![f64::INFINITY; n];
        let mut visited = vec![false; n];
        dist[start] = 0.0;
        for _ in 0..n {
            let mut u = None;
            let mut min_d = f64::INFINITY;
            for v in 0..n { if !visited[v] && dist[v] < min_d { min_d = dist[v]; u = Some(v); } }
            let u = match u { Some(v) => v, None => break };
            visited[u] = true;
            for v in 0..n { if self.adjacency[u][v] {
                let alt = dist[u] + self.weights[u][v];
                if alt < dist[v] { dist[v] = alt; }
            }}
        }
        dist
    }
    pub fn connected_components(&self) -> Vec<Vec<usize>> {
        let n = self.node_count;
        let mut visited = vec![false; n];
        let mut components = Vec::new();
        for start in 0..n {
            if visited[start] { continue; }
            let mut comp = Vec::new();
            let mut stack = vec![start];
            while let Some(v) = stack.pop() {
                if visited[v] { continue; }
                visited[v] = true;
                comp.push(v);
                for w in 0..n {
                    if (self.adjacency[v][w] || self.adjacency[w][v]) && !visited[w] {
                        stack.push(w);
                    }
                }
            }
            components.push(comp);
        }
        components
    }
    pub fn to_dot(&self) -> String {
        let mut out = String::from("digraph {\n");
        for i in 0..self.node_count {
            out.push_str(&format!("  {} [label=\"{}\"];\n", i, self.node_labels[i]));
        }
        for i in 0..self.node_count { for j in 0..self.node_count { if self.adjacency[i][j] {
            out.push_str(&format!("  {} -> {} [label=\"{:.2}\"];\n", i, j, self.weights[i][j]));
        }}}
        out.push_str("}\n");
        out
    }
}

impl fmt::Display for MatrixGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MatrixGraph(n={}, e={})", self.node_count, self.edge_count)
    }
}

/// Cache for matrix computation results.
#[derive(Debug, Clone)]
pub struct MatrixCache {
    entries: Vec<(u64, Vec<f64>)>,
    capacity: usize,
    hits: u64,
    misses: u64,
}

impl MatrixCache {
    pub fn new(capacity: usize) -> Self {
        MatrixCache { entries: Vec::new(), capacity, hits: 0, misses: 0 }
    }
    pub fn get(&mut self, key: u64) -> Option<&Vec<f64>> {
        if let Some(pos) = self.entries.iter().position(|(k, _)| *k == key) {
            self.hits += 1;
            Some(&self.entries[pos].1)
        } else { self.misses += 1; None }
    }
    pub fn insert(&mut self, key: u64, value: Vec<f64>) {
        if self.entries.len() >= self.capacity { self.entries.remove(0); }
        self.entries.push((key, value));
    }
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 { 0.0 } else { self.hits as f64 / total as f64 }
    }
    pub fn size(&self) -> usize { self.entries.len() }
    pub fn clear(&mut self) { self.entries.clear(); self.hits = 0; self.misses = 0; }
}

impl fmt::Display for MatrixCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Cache(size={}, hit_rate={:.1}%)", self.size(), self.hit_rate() * 100.0)
    }
}

/// Compute pairwise distances for matrix elements.
pub fn matrix_pairwise_distances(points: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = points.len();
    let mut distances = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in (i+1)..n {
            let d: f64 = points[i].iter().zip(points[j].iter())
                .map(|(a, b)| (a - b) * (a - b)).sum::<f64>().sqrt();
            distances[i][j] = d;
            distances[j][i] = d;
        }
    }
    distances
}

/// K-means clustering for matrix data.
pub fn matrix_kmeans(data: &[Vec<f64>], k: usize, max_iters: usize) -> Vec<usize> {
    if data.is_empty() || k == 0 { return Vec::new(); }
    let n = data.len();
    let dim = data[0].len();
    let mut centroids: Vec<Vec<f64>> = data.iter().take(k).cloned().collect();
    let mut assignments = vec![0usize; n];
    for _ in 0..max_iters {
        // Assign
        let mut changed = false;
        for i in 0..n {
            let mut best_c = 0; let mut best_d = f64::INFINITY;
            for c in 0..centroids.len() {
                let d: f64 = data[i].iter().zip(centroids[c].iter())
                    .map(|(a, b)| (a - b) * (a - b)).sum();
                if d < best_d { best_d = d; best_c = c; }
            }
            if assignments[i] != best_c { changed = true; assignments[i] = best_c; }
        }
        if !changed { break; }
        // Update centroids
        for c in 0..centroids.len() {
            let members: Vec<usize> = (0..n).filter(|&i| assignments[i] == c).collect();
            if members.is_empty() { continue; }
            for d in 0..dim {
                centroids[c][d] = members.iter().map(|&i| data[i][d]).sum::<f64>() / members.len() as f64;
            }
        }
    }
    assignments
}

/// Principal component analysis (simplified) for matrix data.
pub fn matrix_pca_2d(data: &[Vec<f64>]) -> Vec<(f64, f64)> {
    if data.is_empty() || data[0].len() < 2 { return Vec::new(); }
    let n = data.len();
    let dim = data[0].len();
    // Compute mean
    let mut mean = vec![0.0; dim];
    for row in data { for (j, &v) in row.iter().enumerate() { mean[j] += v; } }
    for j in 0..dim { mean[j] /= n as f64; }
    // Center data
    let centered: Vec<Vec<f64>> = data.iter().map(|row| {
        row.iter().zip(mean.iter()).map(|(v, m)| v - m).collect()
    }).collect();
    // Simple projection onto first two dimensions (not true PCA)
    centered.iter().map(|row| (row[0], row[1])).collect()
}

/// Dense matrix operations for MatRepr computations.
#[derive(Debug, Clone)]
pub struct MatReprDenseMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl MatReprDenseMatrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        MatReprDenseMatrix { rows, cols, data: vec![0.0; rows * cols] }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::new(n, n);
        for i in 0..n { m.data[i * n + i] = 1.0; }
        m
    }

    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), rows * cols);
        MatReprDenseMatrix { rows, cols, data }
    }

    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.cols + j]
    }

    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        self.data[i * self.cols + j] = v;
    }

    pub fn row(&self, i: usize) -> Vec<f64> {
        self.data[i * self.cols..(i + 1) * self.cols].to_vec()
    }

    pub fn col(&self, j: usize) -> Vec<f64> {
        (0..self.rows).map(|i| self.data[i * self.cols + j]).collect()
    }

    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.rows, other.rows); assert_eq!(self.cols, other.cols);
        let data: Vec<f64> = self.data.iter().zip(other.data.iter()).map(|(a, b)| a + b).collect();
        MatReprDenseMatrix { rows: self.rows, cols: self.cols, data }
    }

    pub fn sub(&self, other: &Self) -> Self {
        assert_eq!(self.rows, other.rows); assert_eq!(self.cols, other.cols);
        let data: Vec<f64> = self.data.iter().zip(other.data.iter()).map(|(a, b)| a - b).collect();
        MatReprDenseMatrix { rows: self.rows, cols: self.cols, data }
    }

    pub fn mul_matrix(&self, other: &Self) -> Self {
        assert_eq!(self.cols, other.rows);
        let mut result = Self::new(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols { sum += self.get(i, k) * other.get(k, j); }
                result.set(i, j, sum);
            }
        }
        result
    }

    pub fn scale(&self, s: f64) -> Self {
        let data: Vec<f64> = self.data.iter().map(|&v| v * s).collect();
        MatReprDenseMatrix { rows: self.rows, cols: self.cols, data }
    }

    pub fn transpose(&self) -> Self {
        let mut result = Self::new(self.cols, self.rows);
        for i in 0..self.rows { for j in 0..self.cols { result.set(j, i, self.get(i, j)); } }
        result
    }

    pub fn trace(&self) -> f64 {
        let n = self.rows.min(self.cols);
        (0..n).map(|i| self.get(i, i)).sum()
    }

    pub fn frobenius_norm(&self) -> f64 {
        self.data.iter().map(|&v| v * v).sum::<f64>().sqrt()
    }

    pub fn max_abs(&self) -> f64 {
        self.data.iter().map(|v| v.abs()).fold(0.0f64, f64::max)
    }

    pub fn row_sum(&self, i: usize) -> f64 {
        (0..self.cols).map(|j| self.get(i, j)).sum()
    }

    pub fn col_sum(&self, j: usize) -> f64 {
        (0..self.rows).map(|i| self.get(i, j)).sum()
    }

    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    pub fn is_symmetric(&self) -> bool {
        if !self.is_square() { return false; }
        for i in 0..self.rows { for j in (i+1)..self.cols {
            if (self.get(i, j) - self.get(j, i)).abs() > 1e-10 { return false; }
        }}
        true
    }

    pub fn is_diagonal(&self) -> bool {
        for i in 0..self.rows { for j in 0..self.cols {
            if i != j && self.get(i, j).abs() > 1e-10 { return false; }
        }}
        true
    }

    pub fn is_upper_triangular(&self) -> bool {
        for i in 0..self.rows { for j in 0..i.min(self.cols) {
            if self.get(i, j).abs() > 1e-10 { return false; }
        }}
        true
    }

    pub fn determinant_2x2(&self) -> f64 {
        assert!(self.rows == 2 && self.cols == 2);
        self.get(0, 0) * self.get(1, 1) - self.get(0, 1) * self.get(1, 0)
    }

    pub fn determinant_3x3(&self) -> f64 {
        assert!(self.rows == 3 && self.cols == 3);
        let a = self.get(0, 0); let b = self.get(0, 1); let c = self.get(0, 2);
        let d = self.get(1, 0); let e = self.get(1, 1); let ff = self.get(1, 2);
        let g = self.get(2, 0); let h = self.get(2, 1); let ii = self.get(2, 2);
        a * (e * ii - ff * h) - b * (d * ii - ff * g) + c * (d * h - e * g)
    }

    pub fn inverse_2x2(&self) -> Option<Self> {
        assert!(self.rows == 2 && self.cols == 2);
        let det = self.determinant_2x2();
        if det.abs() < 1e-15 { return None; }
        let inv_det = 1.0 / det;
        let mut result = Self::new(2, 2);
        result.set(0, 0, self.get(1, 1) * inv_det);
        result.set(0, 1, -self.get(0, 1) * inv_det);
        result.set(1, 0, -self.get(1, 0) * inv_det);
        result.set(1, 1, self.get(0, 0) * inv_det);
        Some(result)
    }

    pub fn power(&self, n: u32) -> Self {
        assert!(self.is_square());
        let mut result = Self::identity(self.rows);
        for _ in 0..n { result = result.mul_matrix(self); }
        result
    }

    pub fn submatrix(&self, row_start: usize, col_start: usize, rows: usize, cols: usize) -> Self {
        let mut result = Self::new(rows, cols);
        for i in 0..rows { for j in 0..cols {
            result.set(i, j, self.get(row_start + i, col_start + j));
        }}
        result
    }

    pub fn kronecker_product(&self, other: &Self) -> Self {
        let m = self.rows * other.rows;
        let n = self.cols * other.cols;
        let mut result = Self::new(m, n);
        for i in 0..self.rows { for j in 0..self.cols {
            let s = self.get(i, j);
            for p in 0..other.rows { for q in 0..other.cols {
                result.set(i * other.rows + p, j * other.cols + q, s * other.get(p, q));
            }}
        }}
        result
    }

    pub fn hadamard_product(&self, other: &Self) -> Self {
        assert_eq!(self.rows, other.rows); assert_eq!(self.cols, other.cols);
        let data: Vec<f64> = self.data.iter().zip(other.data.iter()).map(|(a, b)| a * b).collect();
        MatReprDenseMatrix { rows: self.rows, cols: self.cols, data }
    }

    pub fn outer_product(a: &[f64], b: &[f64]) -> Self {
        let mut result = Self::new(a.len(), b.len());
        for i in 0..a.len() { for j in 0..b.len() { result.set(i, j, a[i] * b[j]); } }
        result
    }

    pub fn row_reduce(&self) -> Self {
        let mut result = self.clone();
        let mut pivot_row = 0;
        for col in 0..result.cols {
            if pivot_row >= result.rows { break; }
            let mut max_row = pivot_row;
            for row in (pivot_row + 1)..result.rows {
                if result.get(row, col).abs() > result.get(max_row, col).abs() { max_row = row; }
            }
            if result.get(max_row, col).abs() < 1e-10 { continue; }
            for j in 0..result.cols {
                let tmp = result.get(pivot_row, j);
                result.set(pivot_row, j, result.get(max_row, j));
                result.set(max_row, j, tmp);
            }
            let pivot = result.get(pivot_row, col);
            for j in 0..result.cols { result.set(pivot_row, j, result.get(pivot_row, j) / pivot); }
            for row in 0..result.rows {
                if row == pivot_row { continue; }
                let factor = result.get(row, col);
                for j in 0..result.cols {
                    let v = result.get(row, j) - factor * result.get(pivot_row, j);
                    result.set(row, j, v);
                }
            }
            pivot_row += 1;
        }
        result
    }

    pub fn rank(&self) -> usize {
        let rref = self.row_reduce();
        let mut r = 0;
        for i in 0..rref.rows {
            if (0..rref.cols).any(|j| rref.get(i, j).abs() > 1e-10) { r += 1; }
        }
        r
    }

    pub fn nullity(&self) -> usize {
        self.cols - self.rank()
    }

    pub fn column_space_basis(&self) -> Vec<Vec<f64>> {
        let rref = self.row_reduce();
        let mut basis = Vec::new();
        for j in 0..self.cols {
            let is_pivot = (0..rref.rows).any(|i| {
                (rref.get(i, j) - 1.0).abs() < 1e-10 &&
                (0..j).all(|k| rref.get(i, k).abs() < 1e-10)
            });
            if is_pivot { basis.push(self.col(j)); }
        }
        basis
    }

    pub fn lu_decomposition(&self) -> (Self, Self) {
        assert!(self.is_square());
        let n = self.rows;
        let mut l = Self::identity(n);
        let mut u = self.clone();
        for k in 0..n {
            for i in (k+1)..n {
                if u.get(k, k).abs() < 1e-15 { continue; }
                let factor = u.get(i, k) / u.get(k, k);
                l.set(i, k, factor);
                for j in k..n {
                    let v = u.get(i, j) - factor * u.get(k, j);
                    u.set(i, j, v);
                }
            }
        }
        (l, u)
    }

    pub fn solve(&self, b: &[f64]) -> Option<Vec<f64>> {
        assert!(self.is_square());
        assert_eq!(self.rows, b.len());
        let n = self.rows;
        let mut augmented = Self::new(n, n + 1);
        for i in 0..n { for j in 0..n { augmented.set(i, j, self.get(i, j)); } augmented.set(i, n, b[i]); }
        let rref = augmented.row_reduce();
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            x[i] = rref.get(i, n);
            for j in (i+1)..n { x[i] -= rref.get(i, j) * x[j]; }
            if rref.get(i, i).abs() < 1e-15 { return None; }
            x[i] /= rref.get(i, i);
        }
        Some(x)
    }

    pub fn eigenvalues_2x2(&self) -> (f64, f64) {
        assert!(self.rows == 2 && self.cols == 2);
        let tr = self.trace();
        let det = self.determinant_2x2();
        let disc = tr * tr - 4.0 * det;
        if disc >= 0.0 {
            ((tr + disc.sqrt()) / 2.0, (tr - disc.sqrt()) / 2.0)
        } else {
            (tr / 2.0, tr / 2.0)
        }
    }

    pub fn condition_number(&self) -> f64 {
        let max_sv = self.frobenius_norm();
        if max_sv < 1e-15 { return f64::INFINITY; }
        max_sv
    }

}

impl fmt::Display for MatReprDenseMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MatReprMatrix({}x{})", self.rows, self.cols)
    }
}

/// Interval arithmetic for MatRepr bounds analysis.
#[derive(Debug, Clone, Copy)]
pub struct MatReprInterval {
    pub lo: f64,
    pub hi: f64,
}

impl MatReprInterval {
    pub fn new(lo: f64, hi: f64) -> Self {
        MatReprInterval { lo: lo.min(hi), hi: lo.max(hi) }
    }

    pub fn point(v: f64) -> Self {
        MatReprInterval { lo: v, hi: v }
    }

    pub fn width(&self) -> f64 {
        self.hi - self.lo
    }

    pub fn midpoint(&self) -> f64 {
        (self.lo + self.hi) / 2.0
    }

    pub fn contains(&self, v: f64) -> bool {
        self.lo <= v && v <= self.hi
    }

    pub fn overlaps(&self, other: &Self) -> bool {
        self.lo <= other.hi && other.lo <= self.hi
    }

    pub fn hull(&self, other: &Self) -> Self {
        MatReprInterval { lo: self.lo.min(other.lo), hi: self.hi.max(other.hi) }
    }

    pub fn intersect(&self, other: &Self) -> Option<Self> {
        let lo = self.lo.max(other.lo);
        let hi = self.hi.min(other.hi);
        if lo <= hi { Some(MatReprInterval { lo, hi }) } else { None }
    }

    pub fn add(&self, other: &Self) -> Self {
        MatReprInterval { lo: self.lo + other.lo, hi: self.hi + other.hi }
    }

    pub fn sub(&self, other: &Self) -> Self {
        MatReprInterval { lo: self.lo - other.hi, hi: self.hi - other.lo }
    }

    pub fn mul(&self, other: &Self) -> Self {
        let products = [self.lo * other.lo, self.lo * other.hi, self.hi * other.lo, self.hi * other.hi];
        let lo = products.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = products.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        MatReprInterval { lo, hi }
    }

    pub fn abs(&self) -> Self {
        if self.lo >= 0.0 { *self }
        else if self.hi <= 0.0 { MatReprInterval { lo: -self.hi, hi: -self.lo } }
        else { MatReprInterval { lo: 0.0, hi: self.lo.abs().max(self.hi.abs()) } }
    }

    pub fn sqrt(&self) -> Self {
        let lo = if self.lo >= 0.0 { self.lo.sqrt() } else { 0.0 };
        MatReprInterval { lo, hi: self.hi.max(0.0).sqrt() }
    }

    pub fn is_positive(&self) -> bool {
        self.lo > 0.0
    }

    pub fn is_negative(&self) -> bool {
        self.hi < 0.0
    }

    pub fn is_zero(&self) -> bool {
        self.lo <= 0.0 && self.hi >= 0.0
    }

    pub fn is_point(&self) -> bool {
        (self.hi - self.lo).abs() < 1e-15
    }

}

impl fmt::Display for MatReprInterval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:.4}, {:.4}]", self.lo, self.hi)
    }
}

/// State machine for MatRepr protocol modeling.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MatReprState {
    Uninitialized,
    Loaded,
    Factored,
    Decomposed,
    Verified,
    Invalid,
}

impl fmt::Display for MatReprState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatReprState::Uninitialized => write!(f, "uninitialized"),
            MatReprState::Loaded => write!(f, "loaded"),
            MatReprState::Factored => write!(f, "factored"),
            MatReprState::Decomposed => write!(f, "decomposed"),
            MatReprState::Verified => write!(f, "verified"),
            MatReprState::Invalid => write!(f, "invalid"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MatReprStateMachine {
    pub current: MatReprState,
    pub history: Vec<String>,
    pub transition_count: usize,
}

impl MatReprStateMachine {
    pub fn new() -> Self {
        MatReprStateMachine { current: MatReprState::Uninitialized, history: Vec::new(), transition_count: 0 }
    }
    pub fn state(&self) -> &MatReprState { &self.current }
    pub fn can_transition(&self, target: &MatReprState) -> bool {
        match (&self.current, target) {
            (MatReprState::Uninitialized, MatReprState::Loaded) => true,
            (MatReprState::Loaded, MatReprState::Factored) => true,
            (MatReprState::Loaded, MatReprState::Decomposed) => true,
            (MatReprState::Factored, MatReprState::Verified) => true,
            (MatReprState::Decomposed, MatReprState::Verified) => true,
            (MatReprState::Verified, MatReprState::Loaded) => true,
            (MatReprState::Loaded, MatReprState::Invalid) => true,
            (MatReprState::Invalid, MatReprState::Uninitialized) => true,
            _ => false,
        }
    }
    pub fn transition(&mut self, target: MatReprState) -> bool {
        if self.can_transition(&target) {
            self.history.push(format!("{} -> {}", self.current, target));
            self.current = target;
            self.transition_count += 1;
            true
        } else { false }
    }
    pub fn reset(&mut self) {
        self.current = MatReprState::Uninitialized;
        self.history.clear();
        self.transition_count = 0;
    }
    pub fn history_len(&self) -> usize { self.history.len() }
}

impl fmt::Display for MatReprStateMachine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SM(state={}, transitions={})", self.current, self.transition_count)
    }
}

/// Ring buffer for MatRepr event tracking.
#[derive(Debug, Clone)]
pub struct MatReprRingBuffer {
    data: Vec<f64>,
    capacity: usize,
    head: usize,
    count: usize,
}

impl MatReprRingBuffer {
    pub fn new(capacity: usize) -> Self {
        MatReprRingBuffer { data: vec![0.0; capacity], capacity, head: 0, count: 0 }
    }
    pub fn push(&mut self, value: f64) {
        self.data[self.head] = value;
        self.head = (self.head + 1) % self.capacity;
        if self.count < self.capacity { self.count += 1; }
    }
    pub fn len(&self) -> usize { self.count }
    pub fn is_empty(&self) -> bool { self.count == 0 }
    pub fn is_full(&self) -> bool { self.count == self.capacity }
    pub fn latest(&self) -> Option<f64> {
        if self.count == 0 { None }
        else { Some(self.data[(self.head + self.capacity - 1) % self.capacity]) }
    }
    pub fn oldest(&self) -> Option<f64> {
        if self.count == 0 { None }
        else { Some(self.data[(self.head + self.capacity - self.count) % self.capacity]) }
    }
    pub fn average(&self) -> f64 {
        if self.count == 0 { return 0.0; }
        let mut sum = 0.0;
        for i in 0..self.count {
            sum += self.data[(self.head + self.capacity - 1 - i) % self.capacity];
        }
        sum / self.count as f64
    }
    pub fn to_vec(&self) -> Vec<f64> {
        let mut result = Vec::with_capacity(self.count);
        for i in 0..self.count {
            result.push(self.data[(self.head + self.capacity - self.count + i) % self.capacity]);
        }
        result
    }
    pub fn min(&self) -> Option<f64> {
        if self.count == 0 { return None; }
        Some(self.to_vec().iter().cloned().fold(f64::INFINITY, f64::min))
    }
    pub fn max(&self) -> Option<f64> {
        if self.count == 0 { return None; }
        Some(self.to_vec().iter().cloned().fold(f64::NEG_INFINITY, f64::max))
    }
    pub fn variance(&self) -> f64 {
        if self.count < 2 { return 0.0; }
        let avg = self.average();
        let v: f64 = self.to_vec().iter().map(|&x| (x - avg) * (x - avg)).sum();
        v / (self.count - 1) as f64
    }
    pub fn clear(&mut self) { self.head = 0; self.count = 0; }
}

impl fmt::Display for MatReprRingBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RingBuffer(len={}/{})", self.count, self.capacity)
    }
}

/// Disjoint set (union-find) for MatRepr component tracking.
#[derive(Debug, Clone)]
pub struct MatReprDisjointSet {
    parent: Vec<usize>,
    rank: Vec<usize>,
    size: Vec<usize>,
    num_components: usize,
}

impl MatReprDisjointSet {
    pub fn new(n: usize) -> Self {
        MatReprDisjointSet { parent: (0..n).collect(), rank: vec![0; n], size: vec![1; n], num_components: n }
    }
    pub fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x { self.parent[x] = self.parent[self.parent[x]]; x = self.parent[x]; }
        x
    }
    pub fn union(&mut self, x: usize, y: usize) -> bool {
        let rx = self.find(x); let ry = self.find(y);
        if rx == ry { return false; }
        if self.rank[rx] < self.rank[ry] { self.parent[rx] = ry; self.size[ry] += self.size[rx]; }
        else if self.rank[rx] > self.rank[ry] { self.parent[ry] = rx; self.size[rx] += self.size[ry]; }
        else { self.parent[ry] = rx; self.size[rx] += self.size[ry]; self.rank[rx] += 1; }
        self.num_components -= 1;
        true
    }
    pub fn connected(&mut self, x: usize, y: usize) -> bool { self.find(x) == self.find(y) }
    pub fn component_size(&mut self, x: usize) -> usize { let r = self.find(x); self.size[r] }
    pub fn num_components(&self) -> usize { self.num_components }
    pub fn components(&mut self) -> Vec<Vec<usize>> {
        let n = self.parent.len();
        let mut groups: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
        for i in 0..n { let r = self.find(i); groups.entry(r).or_default().push(i); }
        groups.into_values().collect()
    }
}

impl fmt::Display for MatReprDisjointSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DisjointSet(n={}, components={})", self.parent.len(), self.num_components)
    }
}

/// Sorted list with binary search for MatRepr.
#[derive(Debug, Clone)]
pub struct MatReprSortedList {
    data: Vec<f64>,
}

impl MatReprSortedList {
    pub fn new() -> Self { MatReprSortedList { data: Vec::new() } }
    pub fn insert(&mut self, value: f64) {
        let pos = self.data.partition_point(|&x| x < value);
        self.data.insert(pos, value);
    }
    pub fn contains(&self, value: f64) -> bool {
        self.data.binary_search_by(|x| x.partial_cmp(&value).unwrap()).is_ok()
    }
    pub fn rank(&self, value: f64) -> usize { self.data.partition_point(|&x| x < value) }
    pub fn quantile(&self, q: f64) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let idx = ((self.data.len() - 1) as f64 * q).round() as usize;
        self.data[idx.min(self.data.len() - 1)]
    }
    pub fn len(&self) -> usize { self.data.len() }
    pub fn is_empty(&self) -> bool { self.data.is_empty() }
    pub fn min(&self) -> Option<f64> { self.data.first().copied() }
    pub fn max(&self) -> Option<f64> { self.data.last().copied() }
    pub fn median(&self) -> f64 { self.quantile(0.5) }
    pub fn iqr(&self) -> f64 { self.quantile(0.75) - self.quantile(0.25) }
    pub fn remove(&mut self, value: f64) -> bool {
        if let Ok(pos) = self.data.binary_search_by(|x| x.partial_cmp(&value).unwrap()) {
            self.data.remove(pos); true
        } else { false }
    }
    pub fn range(&self, lo: f64, hi: f64) -> Vec<f64> {
        self.data.iter().filter(|&&x| x >= lo && x <= hi).cloned().collect()
    }
    pub fn to_vec(&self) -> Vec<f64> { self.data.clone() }
}

impl fmt::Display for MatReprSortedList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SortedList(len={})", self.data.len())
    }
}

/// Exponential moving average for MatRepr metrics.
#[derive(Debug, Clone)]
pub struct MatReprEma {
    pub alpha: f64,
    pub value: f64,
    pub count: usize,
    pub initialized: bool,
}

impl MatReprEma {
    pub fn new(alpha: f64) -> Self { MatReprEma { alpha, value: 0.0, count: 0, initialized: false } }
    pub fn update(&mut self, sample: f64) {
        if !self.initialized { self.value = sample; self.initialized = true; }
        else { self.value = self.alpha * sample + (1.0 - self.alpha) * self.value; }
        self.count += 1;
    }
    pub fn current(&self) -> f64 { self.value }
    pub fn reset(&mut self) { self.value = 0.0; self.count = 0; self.initialized = false; }
}

impl fmt::Display for MatReprEma {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EMA(alpha={:.2}, value={:.4})", self.alpha, self.value)
    }
}

/// Simple bloom filter for MatRepr membership testing.
#[derive(Debug, Clone)]
pub struct MatReprBloomFilter {
    bits: Vec<bool>,
    num_hashes: usize,
    size: usize,
    count: usize,
}

impl MatReprBloomFilter {
    pub fn new(size: usize, num_hashes: usize) -> Self {
        MatReprBloomFilter { bits: vec![false; size], num_hashes, size, count: 0 }
    }
    fn hash_indices(&self, value: u64) -> Vec<usize> {
        let mut indices = Vec::with_capacity(self.num_hashes);
        let mut h = value;
        for _ in 0..self.num_hashes {
            h = h.wrapping_mul(0x517cc1b727220a95).wrapping_add(0x6c62272e07bb0142);
            indices.push((h as usize) % self.size);
        }
        indices
    }
    pub fn insert(&mut self, value: u64) {
        for idx in self.hash_indices(value) { self.bits[idx] = true; }
        self.count += 1;
    }
    pub fn may_contain(&self, value: u64) -> bool {
        self.hash_indices(value).iter().all(|&idx| self.bits[idx])
    }
    pub fn false_positive_rate(&self) -> f64 {
        let set_bits = self.bits.iter().filter(|&&b| b).count() as f64;
        (set_bits / self.size as f64).powi(self.num_hashes as i32)
    }
    pub fn count(&self) -> usize { self.count }
    pub fn clear(&mut self) { self.bits.fill(false); self.count = 0; }
}

impl fmt::Display for MatReprBloomFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BloomFilter(size={}, count={}, fpr={:.4})", self.size, self.count, self.false_positive_rate())
    }
}

/// Simple prefix trie for MatRepr string matching.
#[derive(Debug, Clone)]
pub struct MatReprTrieNode {
    children: Vec<(char, usize)>,
    is_terminal: bool,
    value: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct MatReprTrie {
    nodes: Vec<MatReprTrieNode>,
    count: usize,
}

impl MatReprTrie {
    pub fn new() -> Self {
        MatReprTrie { nodes: vec![MatReprTrieNode { children: Vec::new(), is_terminal: false, value: None }], count: 0 }
    }
    pub fn insert(&mut self, key: &str, value: u64) {
        let mut current = 0;
        for ch in key.chars() {
            let next = self.nodes[current].children.iter().find(|(c, _)| *c == ch).map(|(_, idx)| *idx);
            current = match next {
                Some(idx) => idx,
                None => {
                    let idx = self.nodes.len();
                    self.nodes.push(MatReprTrieNode { children: Vec::new(), is_terminal: false, value: None });
                    self.nodes[current].children.push((ch, idx));
                    idx
                }
            };
        }
        self.nodes[current].is_terminal = true;
        self.nodes[current].value = Some(value);
        self.count += 1;
    }
    pub fn search(&self, key: &str) -> Option<u64> {
        let mut current = 0;
        for ch in key.chars() {
            match self.nodes[current].children.iter().find(|(c, _)| *c == ch) {
                Some((_, idx)) => current = *idx,
                None => return None,
            }
        }
        if self.nodes[current].is_terminal { self.nodes[current].value } else { None }
    }
    pub fn starts_with(&self, prefix: &str) -> bool {
        let mut current = 0;
        for ch in prefix.chars() {
            match self.nodes[current].children.iter().find(|(c, _)| *c == ch) {
                Some((_, idx)) => current = *idx,
                None => return false,
            }
        }
        true
    }
    pub fn len(&self) -> usize { self.count }
    pub fn is_empty(&self) -> bool { self.count == 0 }
    pub fn node_count(&self) -> usize { self.nodes.len() }
}

impl fmt::Display for MatReprTrie {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Trie(entries={}, nodes={})", self.count, self.nodes.len())
    }
}

/// Min-heap priority queue for MatRepr scheduling.
#[derive(Debug, Clone)]
pub struct MatReprPriorityQueue {
    heap: Vec<(f64, usize)>,
}

impl MatReprPriorityQueue {
    pub fn new() -> Self { MatReprPriorityQueue { heap: Vec::new() } }
    pub fn push(&mut self, priority: f64, item: usize) {
        self.heap.push((priority, item));
        let mut i = self.heap.len() - 1;
        while i > 0 {
            let parent = (i - 1) / 2;
            if self.heap[i].0 < self.heap[parent].0 { self.heap.swap(i, parent); i = parent; }
            else { break; }
        }
    }
    pub fn pop(&mut self) -> Option<(f64, usize)> {
        if self.heap.is_empty() { return None; }
        let result = self.heap.swap_remove(0);
        if !self.heap.is_empty() { self.sift_down(0); }
        Some(result)
    }
    fn sift_down(&mut self, mut i: usize) {
        loop {
            let left = 2 * i + 1;
            let right = 2 * i + 2;
            let mut smallest = i;
            if left < self.heap.len() && self.heap[left].0 < self.heap[smallest].0 { smallest = left; }
            if right < self.heap.len() && self.heap[right].0 < self.heap[smallest].0 { smallest = right; }
            if smallest != i { self.heap.swap(i, smallest); i = smallest; }
            else { break; }
        }
    }
    pub fn peek(&self) -> Option<&(f64, usize)> { self.heap.first() }
    pub fn len(&self) -> usize { self.heap.len() }
    pub fn is_empty(&self) -> bool { self.heap.is_empty() }
}

impl fmt::Display for MatReprPriorityQueue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PQ(len={})", self.heap.len())
    }
}

/// Online statistics accumulator for MatRepr.
#[derive(Debug, Clone)]
pub struct MatReprAccumulator {
    count: u64,
    mean: f64,
    m2: f64,
    min_val: f64,
    max_val: f64,
    sum: f64,
}

impl MatReprAccumulator {
    pub fn new() -> Self { MatReprAccumulator { count: 0, mean: 0.0, m2: 0.0, min_val: f64::INFINITY, max_val: f64::NEG_INFINITY, sum: 0.0 } }
    pub fn add(&mut self, value: f64) {
        self.count += 1;
        self.sum += value;
        self.min_val = self.min_val.min(value);
        self.max_val = self.max_val.max(value);
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }
    pub fn count(&self) -> u64 { self.count }
    pub fn mean(&self) -> f64 { self.mean }
    pub fn variance(&self) -> f64 { if self.count < 2 { 0.0 } else { self.m2 / (self.count - 1) as f64 } }
    pub fn std_dev(&self) -> f64 { self.variance().sqrt() }
    pub fn min(&self) -> f64 { self.min_val }
    pub fn max(&self) -> f64 { self.max_val }
    pub fn sum(&self) -> f64 { self.sum }
    pub fn range(&self) -> f64 { self.max_val - self.min_val }
    pub fn coefficient_of_variation(&self) -> f64 {
        if self.mean.abs() < 1e-15 { 0.0 } else { self.std_dev() / self.mean.abs() }
    }
    pub fn merge(&mut self, other: &Self) {
        if other.count == 0 { return; }
        let total = self.count + other.count;
        let delta = other.mean - self.mean;
        let new_mean = (self.sum + other.sum) / total as f64;
        self.m2 += other.m2 + delta * delta * (self.count as f64 * other.count as f64 / total as f64);
        self.mean = new_mean;
        self.count = total;
        self.sum += other.sum;
        self.min_val = self.min_val.min(other.min_val);
        self.max_val = self.max_val.max(other.max_val);
    }
    pub fn reset(&mut self) {
        self.count = 0; self.mean = 0.0; self.m2 = 0.0;
        self.min_val = f64::INFINITY; self.max_val = f64::NEG_INFINITY; self.sum = 0.0;
    }
}

impl fmt::Display for MatReprAccumulator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Acc(n={}, mean={:.4}, std={:.4})", self.count, self.mean, self.std_dev())
    }
}

/// Sparse matrix (COO format) for MatRepr.
#[derive(Debug, Clone)]
pub struct MatReprSparseMatrix {
    pub rows: usize,
    pub cols: usize,
    pub entries: Vec<(usize, usize, f64)>,
}

impl MatReprSparseMatrix {
    pub fn new(rows: usize, cols: usize) -> Self { MatReprSparseMatrix { rows, cols, entries: Vec::new() } }
    pub fn insert(&mut self, i: usize, j: usize, v: f64) {
        if let Some(pos) = self.entries.iter().position(|&(r, c, _)| r == i && c == j) {
            self.entries[pos].2 = v;
        } else { self.entries.push((i, j, v)); }
    }
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.entries.iter().find(|&&(r, c, _)| r == i && c == j).map(|&(_, _, v)| v).unwrap_or(0.0)
    }
    pub fn nnz(&self) -> usize { self.entries.len() }
    pub fn density(&self) -> f64 { self.entries.len() as f64 / (self.rows * self.cols) as f64 }
    pub fn transpose(&self) -> Self {
        let mut result = MatReprSparseMatrix::new(self.cols, self.rows);
        for &(i, j, v) in &self.entries { result.entries.push((j, i, v)); }
        result
    }
    pub fn add(&self, other: &Self) -> Self {
        let mut result = MatReprSparseMatrix::new(self.rows, self.cols);
        for &(i, j, v) in &self.entries { result.insert(i, j, result.get(i, j) + v); }
        for &(i, j, v) in &other.entries { result.insert(i, j, result.get(i, j) + v); }
        result
    }
    pub fn scale(&self, s: f64) -> Self {
        let mut result = MatReprSparseMatrix::new(self.rows, self.cols);
        for &(i, j, v) in &self.entries { result.entries.push((i, j, v * s)); }
        result
    }
    pub fn mul_vec(&self, x: &[f64]) -> Vec<f64> {
        let mut result = vec![0.0; self.rows];
        for &(i, j, v) in &self.entries { result[i] += v * x[j]; }
        result
    }
    pub fn frobenius_norm(&self) -> f64 { self.entries.iter().map(|&(_, _, v)| v * v).sum::<f64>().sqrt() }
    pub fn row_nnz(&self, i: usize) -> usize { self.entries.iter().filter(|&&(r, _, _)| r == i).count() }
    pub fn col_nnz(&self, j: usize) -> usize { self.entries.iter().filter(|&&(_, c, _)| c == j).count() }
    pub fn to_dense(&self, dm_new: fn(usize, usize) -> Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let mut result = vec![vec![0.0; self.cols]; self.rows];
        for &(i, j, v) in &self.entries { result[i][j] = v; }
        result
    }
    pub fn diagonal(&self) -> Vec<f64> {
        let n = self.rows.min(self.cols);
        (0..n).map(|i| self.get(i, i)).collect()
    }
    pub fn trace(&self) -> f64 { self.diagonal().iter().sum() }
    pub fn remove_zeros(&mut self, tol: f64) {
        self.entries.retain(|&(_, _, v)| v.abs() > tol);
    }
}

impl fmt::Display for MatReprSparseMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Sparse({}x{}, nnz={})", self.rows, self.cols, self.nnz())
    }
}

/// Polynomial with f64 coefficients for MatRepr.
#[derive(Debug, Clone)]
pub struct MatReprPolynomial {
    pub coefficients: Vec<f64>,
}

impl MatReprPolynomial {
    pub fn new(coeffs: Vec<f64>) -> Self { MatReprPolynomial { coefficients: coeffs } }
    pub fn zero() -> Self { MatReprPolynomial { coefficients: vec![0.0] } }
    pub fn one() -> Self { MatReprPolynomial { coefficients: vec![1.0] } }
    pub fn monomial(degree: usize, coeff: f64) -> Self {
        let mut c = vec![0.0; degree + 1];
        c[degree] = coeff;
        MatReprPolynomial { coefficients: c }
    }
    pub fn degree(&self) -> usize {
        if self.coefficients.is_empty() { return 0; }
        let mut d = self.coefficients.len() - 1;
        while d > 0 && self.coefficients[d].abs() < 1e-15 { d -= 1; }
        d
    }
    pub fn evaluate(&self, x: f64) -> f64 {
        let mut result = 0.0;
        let mut power = 1.0;
        for &c in &self.coefficients {
            result += c * power;
            power *= x;
        }
        result
    }
    pub fn evaluate_horner(&self, x: f64) -> f64 {
        if self.coefficients.is_empty() { return 0.0; }
        let mut result = *self.coefficients.last().unwrap();
        for &c in self.coefficients.iter().rev().skip(1) {
            result = result * x + c;
        }
        result
    }
    pub fn add(&self, other: &Self) -> Self {
        let n = self.coefficients.len().max(other.coefficients.len());
        let mut result = vec![0.0; n];
        for (i, &c) in self.coefficients.iter().enumerate() { result[i] += c; }
        for (i, &c) in other.coefficients.iter().enumerate() { result[i] += c; }
        MatReprPolynomial { coefficients: result }
    }
    pub fn sub(&self, other: &Self) -> Self {
        let n = self.coefficients.len().max(other.coefficients.len());
        let mut result = vec![0.0; n];
        for (i, &c) in self.coefficients.iter().enumerate() { result[i] += c; }
        for (i, &c) in other.coefficients.iter().enumerate() { result[i] -= c; }
        MatReprPolynomial { coefficients: result }
    }
    pub fn mul(&self, other: &Self) -> Self {
        let n = self.coefficients.len() + other.coefficients.len() - 1;
        let mut result = vec![0.0; n];
        for (i, &a) in self.coefficients.iter().enumerate() {
            for (j, &b) in other.coefficients.iter().enumerate() {
                result[i + j] += a * b;
            }
        }
        MatReprPolynomial { coefficients: result }
    }
    pub fn scale(&self, s: f64) -> Self {
        MatReprPolynomial { coefficients: self.coefficients.iter().map(|&c| c * s).collect() }
    }
    pub fn derivative(&self) -> Self {
        if self.coefficients.len() <= 1 { return Self::zero(); }
        let coeffs: Vec<f64> = self.coefficients.iter().enumerate().skip(1)
            .map(|(i, &c)| c * i as f64).collect();
        MatReprPolynomial { coefficients: coeffs }
    }
    pub fn integral(&self, constant: f64) -> Self {
        let mut coeffs = vec![constant];
        for (i, &c) in self.coefficients.iter().enumerate() {
            coeffs.push(c / (i + 1) as f64);
        }
        MatReprPolynomial { coefficients: coeffs }
    }
    pub fn roots_quadratic(&self) -> Vec<f64> {
        if self.degree() != 2 { return Vec::new(); }
        let a = self.coefficients[2];
        let b = self.coefficients[1];
        let c = self.coefficients[0];
        let disc = b * b - 4.0 * a * c;
        if disc < 0.0 { Vec::new() }
        else if disc.abs() < 1e-15 { vec![-b / (2.0 * a)] }
        else { vec![(-b + disc.sqrt()) / (2.0 * a), (-b - disc.sqrt()) / (2.0 * a)] }
    }
    pub fn is_zero(&self) -> bool { self.coefficients.iter().all(|&c| c.abs() < 1e-15) }
    pub fn leading_coefficient(&self) -> f64 {
        self.coefficients.get(self.degree()).copied().unwrap_or(0.0)
    }
    pub fn compose(&self, other: &Self) -> Self {
        let mut result = Self::zero();
        let mut power = Self::one();
        for &c in &self.coefficients {
            result = result.add(&power.scale(c));
            power = power.mul(other);
        }
        result
    }
    pub fn newton_root(&self, initial_guess: f64, max_iters: usize, tol: f64) -> Option<f64> {
        let deriv = self.derivative();
        let mut x = initial_guess;
        for _ in 0..max_iters {
            let fx = self.evaluate(x);
            if fx.abs() < tol { return Some(x); }
            let dfx = deriv.evaluate(x);
            if dfx.abs() < 1e-15 { return None; }
            x -= fx / dfx;
        }
        if self.evaluate(x).abs() < tol * 100.0 { Some(x) } else { None }
    }
}

impl fmt::Display for MatReprPolynomial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut terms = Vec::new();
        for (i, &c) in self.coefficients.iter().enumerate() {
            if c.abs() < 1e-15 { continue; }
            if i == 0 { terms.push(format!("{:.2}", c)); }
            else if i == 1 { terms.push(format!("{:.2}x", c)); }
            else { terms.push(format!("{:.2}x^{}", c, i)); }
        }
        if terms.is_empty() { write!(f, "0") }
        else { write!(f, "{}", terms.join(" + ")) }
    }
}

/// Simple linear congruential generator for MatRepr.
#[derive(Debug, Clone)]
pub struct MatReprRng {
    state: u64,
}

impl MatReprRng {
    pub fn new(seed: u64) -> Self { MatReprRng { state: seed } }
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }
    pub fn next_f64(&mut self) -> f64 { (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 }
    pub fn next_range(&mut self, lo: u64, hi: u64) -> u64 {
        if hi <= lo { return lo; }
        lo + (self.next_u64() % (hi - lo))
    }
    pub fn next_gaussian(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    pub fn shuffle(&mut self, data: &mut [f64]) {
        let n = data.len();
        for i in (1..n).rev() {
            let j = self.next_range(0, i as u64 + 1) as usize;
            data.swap(i, j);
        }
    }
    pub fn sample(&mut self, data: &[f64], n: usize) -> Vec<f64> {
        let mut result = Vec::with_capacity(n);
        for _ in 0..n {
            let idx = self.next_range(0, data.len() as u64) as usize;
            result.push(data[idx]);
        }
        result
    }
    pub fn bernoulli(&mut self, p: f64) -> bool { self.next_f64() < p }
    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 { lo + self.next_f64() * (hi - lo) }
    pub fn exponential(&mut self, lambda: f64) -> f64 { -self.next_f64().max(1e-15).ln() / lambda }
}

impl fmt::Display for MatReprRng {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Rng(state={:#x})", self.state)
    }
}

/// Simple timing utility for MatRepr benchmarking.
#[derive(Debug, Clone)]
pub struct MatReprTimer {
    pub label: String,
    pub elapsed_ns: Vec<u64>,
    pub running: bool,
}

impl MatReprTimer {
    pub fn new(label: impl Into<String>) -> Self { MatReprTimer { label: label.into(), elapsed_ns: Vec::new(), running: false } }
    pub fn record(&mut self, ns: u64) { self.elapsed_ns.push(ns); }
    pub fn total_ns(&self) -> u64 { self.elapsed_ns.iter().sum() }
    pub fn count(&self) -> usize { self.elapsed_ns.len() }
    pub fn average_ns(&self) -> f64 {
        if self.elapsed_ns.is_empty() { 0.0 } else { self.total_ns() as f64 / self.elapsed_ns.len() as f64 }
    }
    pub fn min_ns(&self) -> u64 { self.elapsed_ns.iter().cloned().min().unwrap_or(0) }
    pub fn max_ns(&self) -> u64 { self.elapsed_ns.iter().cloned().max().unwrap_or(0) }
    pub fn percentile_ns(&self, p: f64) -> u64 {
        if self.elapsed_ns.is_empty() { return 0; }
        let mut sorted = self.elapsed_ns.clone();
        sorted.sort();
        let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    }
    pub fn p50_ns(&self) -> u64 { self.percentile_ns(0.5) }
    pub fn p95_ns(&self) -> u64 { self.percentile_ns(0.95) }
    pub fn p99_ns(&self) -> u64 { self.percentile_ns(0.99) }
    pub fn reset(&mut self) { self.elapsed_ns.clear(); self.running = false; }
}

impl fmt::Display for MatReprTimer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Timer({}: avg={:.0}ns, n={})", self.label, self.average_ns(), self.count())
    }
}

/// Compact bit vector for MatRepr set operations.
#[derive(Debug, Clone)]
pub struct MatReprBitVector {
    words: Vec<u64>,
    len: usize,
}

impl MatReprBitVector {
    pub fn new(len: usize) -> Self { MatReprBitVector { words: vec![0u64; (len + 63) / 64], len } }
    pub fn set(&mut self, i: usize) { if i < self.len { self.words[i / 64] |= 1u64 << (i % 64); } }
    pub fn clear(&mut self, i: usize) { if i < self.len { self.words[i / 64] &= !(1u64 << (i % 64)); } }
    pub fn get(&self, i: usize) -> bool { i < self.len && (self.words[i / 64] & (1u64 << (i % 64))) != 0 }
    pub fn len(&self) -> usize { self.len }
    pub fn count_ones(&self) -> usize { self.words.iter().map(|w| w.count_ones() as usize).sum() }
    pub fn count_zeros(&self) -> usize { self.len - self.count_ones() }
    pub fn is_empty(&self) -> bool { self.count_ones() == 0 }
    pub fn and(&self, other: &Self) -> Self {
        let n = self.words.len().min(other.words.len());
        let mut result = Self::new(self.len.min(other.len));
        for i in 0..n { result.words[i] = self.words[i] & other.words[i]; }
        result
    }
    pub fn or(&self, other: &Self) -> Self {
        let n = self.words.len().max(other.words.len());
        let mut result = Self::new(self.len.max(other.len));
        for i in 0..self.words.len().min(n) { result.words[i] |= self.words[i]; }
        for i in 0..other.words.len().min(n) { result.words[i] |= other.words[i]; }
        result
    }
    pub fn xor(&self, other: &Self) -> Self {
        let n = self.words.len().max(other.words.len());
        let mut result = Self::new(self.len.max(other.len));
        for i in 0..self.words.len().min(n) { result.words[i] = self.words[i]; }
        for i in 0..other.words.len().min(n) { result.words[i] ^= other.words[i]; }
        result
    }
    pub fn not(&self) -> Self {
        let mut result = Self::new(self.len);
        for i in 0..self.words.len() { result.words[i] = !self.words[i]; }
        // Clear unused bits in last word
        let extra = self.len % 64;
        if extra > 0 && !result.words.is_empty() {
            let last = result.words.len() - 1;
            result.words[last] &= (1u64 << extra) - 1;
        }
        result
    }
    pub fn iter_ones(&self) -> Vec<usize> {
        let mut result = Vec::new();
        for i in 0..self.len { if self.get(i) { result.push(i); } }
        result
    }
    pub fn jaccard(&self, other: &Self) -> f64 {
        let intersection = self.and(other).count_ones() as f64;
        let union = self.or(other).count_ones() as f64;
        if union == 0.0 { 1.0 } else { intersection / union }
    }
    pub fn hamming_distance(&self, other: &Self) -> usize { self.xor(other).count_ones() }
    pub fn fill(&mut self, value: bool) {
        let fill_val = if value { u64::MAX } else { 0 };
        for w in &mut self.words { *w = fill_val; }
        if value { let extra = self.len % 64; if extra > 0 { let last = self.words.len() - 1; self.words[last] &= (1u64 << extra) - 1; } }
    }
}

impl fmt::Display for MatReprBitVector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BitVec(len={}, ones={})", self.len, self.count_ones())
    }
}

/// LRU cache for MatRepr computation memoization.
#[derive(Debug, Clone)]
pub struct MatReprLruCache {
    entries: Vec<(u64, Vec<f64>, u64)>,
    capacity: usize,
    clock: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
}

impl MatReprLruCache {
    pub fn new(capacity: usize) -> Self { MatReprLruCache { entries: Vec::new(), capacity, clock: 0, hits: 0, misses: 0, evictions: 0 } }
    pub fn get(&mut self, key: u64) -> Option<&Vec<f64>> {
        self.clock += 1;
        if let Some(pos) = self.entries.iter().position(|(k, _, _)| *k == key) {
            self.entries[pos].2 = self.clock;
            self.hits += 1;
            Some(&self.entries[pos].1)
        } else { self.misses += 1; None }
    }
    pub fn put(&mut self, key: u64, value: Vec<f64>) {
        self.clock += 1;
        if let Some(pos) = self.entries.iter().position(|(k, _, _)| *k == key) {
            self.entries[pos].1 = value;
            self.entries[pos].2 = self.clock;
            return;
        }
        if self.entries.len() >= self.capacity {
            let lru_pos = self.entries.iter().enumerate()
                .min_by_key(|(_, (_, _, ts))| *ts).map(|(i, _)| i).unwrap();
            self.entries.remove(lru_pos);
            self.evictions += 1;
        }
        self.entries.push((key, value, self.clock));
    }
    pub fn size(&self) -> usize { self.entries.len() }
    pub fn hit_rate(&self) -> f64 { let t = self.hits + self.misses; if t == 0 { 0.0 } else { self.hits as f64 / t as f64 } }
    pub fn eviction_count(&self) -> u64 { self.evictions }
    pub fn contains(&self, key: u64) -> bool { self.entries.iter().any(|(k, _, _)| *k == key) }
    pub fn clear(&mut self) { self.entries.clear(); self.hits = 0; self.misses = 0; self.evictions = 0; self.clock = 0; }
    pub fn keys(&self) -> Vec<u64> { self.entries.iter().map(|(k, _, _)| *k).collect() }
}

impl fmt::Display for MatReprLruCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LRU(size={}/{}, hr={:.1}%)", self.size(), self.capacity, self.hit_rate() * 100.0)
    }
}

/// Graph coloring utility for MatRepr scheduling.
#[derive(Debug, Clone)]
pub struct MatReprGraphColoring {
    pub adjacency: Vec<Vec<bool>>,
    pub colors: Vec<Option<usize>>,
    pub num_nodes: usize,
    pub num_colors_used: usize,
}

impl MatReprGraphColoring {
    pub fn new(n: usize) -> Self {
        MatReprGraphColoring { adjacency: vec![vec![false; n]; n], colors: vec![None; n], num_nodes: n, num_colors_used: 0 }
    }
    pub fn add_edge(&mut self, i: usize, j: usize) {
        if i < self.num_nodes && j < self.num_nodes {
            self.adjacency[i][j] = true;
            self.adjacency[j][i] = true;
        }
    }
    pub fn greedy_color(&mut self) -> usize {
        self.colors = vec![None; self.num_nodes];
        let mut max_color = 0;
        for v in 0..self.num_nodes {
            let neighbor_colors: std::collections::HashSet<usize> = (0..self.num_nodes)
                .filter(|&u| self.adjacency[v][u] && self.colors[u].is_some())
                .map(|u| self.colors[u].unwrap()).collect();
            let mut c = 0;
            while neighbor_colors.contains(&c) { c += 1; }
            self.colors[v] = Some(c);
            max_color = max_color.max(c);
        }
        self.num_colors_used = max_color + 1;
        self.num_colors_used
    }
    pub fn is_valid_coloring(&self) -> bool {
        for i in 0..self.num_nodes {
            for j in (i+1)..self.num_nodes {
                if self.adjacency[i][j] {
                    if let (Some(ci), Some(cj)) = (self.colors[i], self.colors[j]) {
                        if ci == cj { return false; }
                    }
                }
            }
        }
        true
    }
    pub fn chromatic_number_upper_bound(&self) -> usize {
        let max_degree = (0..self.num_nodes)
            .map(|v| (0..self.num_nodes).filter(|&u| self.adjacency[v][u]).count())
            .max().unwrap_or(0);
        max_degree + 1
    }
    pub fn color_classes(&self) -> Vec<Vec<usize>> {
        let mut classes: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
        for (v, &c) in self.colors.iter().enumerate() {
            if let Some(color) = c { classes.entry(color).or_default().push(v); }
        }
        let mut result: Vec<Vec<usize>> = classes.into_values().collect();
        result.sort_by_key(|v| v[0]);
        result
    }
}

impl fmt::Display for MatReprGraphColoring {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Coloring(n={}, colors={})", self.num_nodes, self.num_colors_used)
    }
}

/// Top-K tracker for MatRepr ranking.
#[derive(Debug, Clone)]
pub struct MatReprTopK {
    pub k: usize,
    pub items: Vec<(f64, String)>,
}

impl MatReprTopK {
    pub fn new(k: usize) -> Self { MatReprTopK { k, items: Vec::new() } }
    pub fn insert(&mut self, score: f64, label: impl Into<String>) {
        self.items.push((score, label.into()));
        self.items.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        if self.items.len() > self.k { self.items.truncate(self.k); }
    }
    pub fn top(&self) -> &[(f64, String)] { &self.items }
    pub fn min_score(&self) -> Option<f64> { self.items.last().map(|(s, _)| *s) }
    pub fn max_score(&self) -> Option<f64> { self.items.first().map(|(s, _)| *s) }
    pub fn is_full(&self) -> bool { self.items.len() >= self.k }
    pub fn len(&self) -> usize { self.items.len() }
    pub fn contains_label(&self, label: &str) -> bool { self.items.iter().any(|(_, l)| l == label) }
    pub fn clear(&mut self) { self.items.clear(); }
    pub fn merge(&mut self, other: &Self) {
        for (score, label) in &other.items { self.insert(*score, label.clone()); }
    }
}

impl fmt::Display for MatReprTopK {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TopK(k={}, len={})", self.k, self.items.len())
    }
}

/// Sliding window statistics for MatRepr monitoring.
#[derive(Debug, Clone)]
pub struct MatReprSlidingWindow {
    pub window_size: usize,
    pub data: Vec<f64>,
    pub sum: f64,
}

impl MatReprSlidingWindow {
    pub fn new(window_size: usize) -> Self { MatReprSlidingWindow { window_size, data: Vec::new(), sum: 0.0 } }
    pub fn push(&mut self, value: f64) {
        self.data.push(value);
        self.sum += value;
        if self.data.len() > self.window_size {
            self.sum -= self.data.remove(0);
        }
    }
    pub fn mean(&self) -> f64 { if self.data.is_empty() { 0.0 } else { self.sum / self.data.len() as f64 } }
    pub fn variance(&self) -> f64 {
        if self.data.len() < 2 { return 0.0; }
        let m = self.mean();
        self.data.iter().map(|&x| (x - m) * (x - m)).sum::<f64>() / (self.data.len() - 1) as f64
    }
    pub fn std_dev(&self) -> f64 { self.variance().sqrt() }
    pub fn min(&self) -> f64 { self.data.iter().cloned().fold(f64::INFINITY, f64::min) }
    pub fn max(&self) -> f64 { self.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max) }
    pub fn len(&self) -> usize { self.data.len() }
    pub fn is_full(&self) -> bool { self.data.len() >= self.window_size }
    pub fn trend(&self) -> f64 {
        if self.data.len() < 2 { return 0.0; }
        let n = self.data.len() as f64;
        let sum_x: f64 = (0..self.data.len()).map(|i| i as f64).sum();
        let sum_y: f64 = self.data.iter().sum();
        let sum_xy: f64 = self.data.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
        let sum_xx: f64 = (0..self.data.len()).map(|i| (i as f64) * (i as f64)).sum();
        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-15 { 0.0 } else { (n * sum_xy - sum_x * sum_y) / denom }
    }
    pub fn anomaly_score(&self, value: f64) -> f64 {
        let s = self.std_dev();
        if s.abs() < 1e-15 { return 0.0; }
        ((value - self.mean()) / s).abs()
    }
    pub fn clear(&mut self) { self.data.clear(); self.sum = 0.0; }
}

impl fmt::Display for MatReprSlidingWindow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Window(size={}/{}, mean={:.2})", self.data.len(), self.window_size, self.mean())
    }
}

/// Confusion matrix for MatRepr classification evaluation.
#[derive(Debug, Clone)]
pub struct MatReprConfusionMatrix {
    pub true_positive: u64,
    pub false_positive: u64,
    pub true_negative: u64,
    pub false_negative: u64,
}

impl MatReprConfusionMatrix {
    pub fn new() -> Self { MatReprConfusionMatrix { true_positive: 0, false_positive: 0, true_negative: 0, false_negative: 0 } }
    pub fn from_predictions(actual: &[bool], predicted: &[bool]) -> Self {
        let mut cm = Self::new();
        for (&a, &p) in actual.iter().zip(predicted.iter()) {
            match (a, p) {
                (true, true) => cm.true_positive += 1,
                (false, true) => cm.false_positive += 1,
                (true, false) => cm.false_negative += 1,
                (false, false) => cm.true_negative += 1,
            }
        }
        cm
    }
    pub fn total(&self) -> u64 { self.true_positive + self.false_positive + self.true_negative + self.false_negative }
    pub fn accuracy(&self) -> f64 { let t = self.total(); if t == 0 { 0.0 } else { (self.true_positive + self.true_negative) as f64 / t as f64 } }
    pub fn precision(&self) -> f64 { let d = self.true_positive + self.false_positive; if d == 0 { 0.0 } else { self.true_positive as f64 / d as f64 } }
    pub fn recall(&self) -> f64 { let d = self.true_positive + self.false_negative; if d == 0 { 0.0 } else { self.true_positive as f64 / d as f64 } }
    pub fn f1_score(&self) -> f64 { let p = self.precision(); let r = self.recall(); if p + r == 0.0 { 0.0 } else { 2.0 * p * r / (p + r) } }
    pub fn specificity(&self) -> f64 { let d = self.true_negative + self.false_positive; if d == 0 { 0.0 } else { self.true_negative as f64 / d as f64 } }
    pub fn false_positive_rate(&self) -> f64 { 1.0 - self.specificity() }
    pub fn matthews_correlation(&self) -> f64 {
        let tp = self.true_positive as f64; let fp = self.false_positive as f64;
        let tn = self.true_negative as f64; let fn_ = self.false_negative as f64;
        let num = tp * tn - fp * fn_;
        let den = ((tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_)).sqrt();
        if den == 0.0 { 0.0 } else { num / den }
    }
    pub fn merge(&mut self, other: &Self) {
        self.true_positive += other.true_positive;
        self.false_positive += other.false_positive;
        self.true_negative += other.true_negative;
        self.false_negative += other.false_negative;
    }
}

impl fmt::Display for MatReprConfusionMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CM(acc={:.3}, prec={:.3}, rec={:.3}, f1={:.3})",
            self.accuracy(), self.precision(), self.recall(), self.f1_score())
    }
}

/// Cosine similarity for MatRepr feature vectors.
pub fn matrepr_cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na * nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

/// Euclidean distance for MatRepr.
pub fn matrepr_euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt()
}

/// Manhattan distance for MatRepr.
pub fn matrepr_manhattan_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

/// Chebyshev distance for MatRepr.
pub fn matrepr_chebyshev_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f64, f64::max)
}

/// Minkowski distance for MatRepr.
pub fn matrepr_minkowski_distance(a: &[f64], b: &[f64], p: f64) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs().powf(p)).sum::<f64>().powf(1.0 / p)
}

/// Normalize a vector for MatRepr.
pub fn matrepr_normalize(v: &[f64]) -> Vec<f64> {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm == 0.0 { v.to_vec() } else { v.iter().map(|x| x / norm).collect() }
}

/// Dot product for MatRepr.
pub fn matrepr_dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Cross product (3D) for MatRepr.
pub fn matrepr_cross_product(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]
}

/// Linear interpolation for MatRepr.
pub fn matrepr_lerp(a: f64, b: f64, t: f64) -> f64 { a + (b - a) * t }

/// Clamp value for MatRepr.
pub fn matrepr_clamp(v: f64, lo: f64, hi: f64) -> f64 { v.max(lo).min(hi) }

/// Sigmoid function for MatRepr.
pub fn matrepr_sigmoid(x: f64) -> f64 { 1.0 / (1.0 + (-x).exp()) }

/// Softmax for MatRepr.
pub fn matrepr_softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// Log-sum-exp for MatRepr.
pub fn matrepr_logsumexp(values: &[f64]) -> f64 {
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    max + values.iter().map(|&v| (v - max).exp()).sum::<f64>().ln()
}

/// KL divergence for MatRepr.
pub fn matrepr_kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    p.iter().zip(q.iter()).map(|(&pi, &qi)| {
        if pi > 0.0 && qi > 0.0 { pi * (pi / qi).ln() } else { 0.0 }
    }).sum()
}

/// Jensen-Shannon divergence for MatRepr.
pub fn matrepr_js_divergence(p: &[f64], q: &[f64]) -> f64 {
    let m: Vec<f64> = p.iter().zip(q.iter()).map(|(&pi, &qi)| (pi + qi) / 2.0).collect();
    (matrepr_kl_divergence(p, &m) + matrepr_kl_divergence(q, &m)) / 2.0
}

/// Total variation distance for MatRepr.
pub fn matrepr_tv_distance(p: &[f64], q: &[f64]) -> f64 {
    p.iter().zip(q.iter()).map(|(&pi, &qi)| (pi - qi).abs()).sum::<f64>() / 2.0
}

/// Hellinger distance for MatRepr.
pub fn matrepr_hellinger_distance(p: &[f64], q: &[f64]) -> f64 {
    let sum: f64 = p.iter().zip(q.iter()).map(|(&pi, &qi)| {
        let diff = pi.sqrt() - qi.sqrt();
        diff * diff
    }).sum();
    (sum / 2.0).sqrt()
}

/// Earth mover's distance (1D) for MatRepr.
pub fn matrepr_emd_1d(p: &[f64], q: &[f64]) -> f64 {
    let mut cum_diff = 0.0;
    let mut total = 0.0;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        cum_diff += pi - qi;
        total += cum_diff.abs();
    }
    total
}

/// Feature scaling utilities for MatRepr.
#[derive(Debug, Clone)]
pub struct MatReprFeatureScaler {
    pub means: Vec<f64>,
    pub stds: Vec<f64>,
    pub mins: Vec<f64>,
    pub maxs: Vec<f64>,
    pub fitted: bool,
}

impl MatReprFeatureScaler {
    pub fn new() -> Self { MatReprFeatureScaler { means: Vec::new(), stds: Vec::new(), mins: Vec::new(), maxs: Vec::new(), fitted: false } }
    pub fn fit(&mut self, data: &[Vec<f64>]) {
        if data.is_empty() { return; }
        let dim = data[0].len();
        let n = data.len() as f64;
        self.means = vec![0.0; dim];
        self.mins = vec![f64::INFINITY; dim];
        self.maxs = vec![f64::NEG_INFINITY; dim];
        for row in data {
            for (j, &v) in row.iter().enumerate() {
                self.means[j] += v;
                self.mins[j] = self.mins[j].min(v);
                self.maxs[j] = self.maxs[j].max(v);
            }
        }
        for j in 0..dim { self.means[j] /= n; }
        self.stds = vec![0.0; dim];
        for row in data {
            for (j, &v) in row.iter().enumerate() {
                self.stds[j] += (v - self.means[j]).powi(2);
            }
        }
        for j in 0..dim { self.stds[j] = (self.stds[j] / (n - 1.0).max(1.0)).sqrt(); }
        self.fitted = true;
    }
    pub fn standardize(&self, row: &[f64]) -> Vec<f64> {
        row.iter().enumerate().map(|(j, &v)| {
            if self.stds[j].abs() < 1e-15 { 0.0 } else { (v - self.means[j]) / self.stds[j] }
        }).collect()
    }
    pub fn normalize(&self, row: &[f64]) -> Vec<f64> {
        row.iter().enumerate().map(|(j, &v)| {
            let range = self.maxs[j] - self.mins[j];
            if range.abs() < 1e-15 { 0.0 } else { (v - self.mins[j]) / range }
        }).collect()
    }
    pub fn inverse_standardize(&self, row: &[f64]) -> Vec<f64> {
        row.iter().enumerate().map(|(j, &v)| v * self.stds[j] + self.means[j]).collect()
    }
    pub fn inverse_normalize(&self, row: &[f64]) -> Vec<f64> {
        row.iter().enumerate().map(|(j, &v)| v * (self.maxs[j] - self.mins[j]) + self.mins[j]).collect()
    }
    pub fn dimension(&self) -> usize { self.means.len() }
}

impl fmt::Display for MatReprFeatureScaler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Scaler(dim={}, fitted={})", self.dimension(), self.fitted)
    }
}

/// Simple linear regression for MatRepr trend analysis.
#[derive(Debug, Clone)]
pub struct MatReprLinearRegression {
    pub slope: f64,
    pub intercept: f64,
    pub r_squared: f64,
    pub fitted: bool,
}

impl MatReprLinearRegression {
    pub fn new() -> Self { MatReprLinearRegression { slope: 0.0, intercept: 0.0, r_squared: 0.0, fitted: false } }
    pub fn fit(&mut self, x: &[f64], y: &[f64]) {
        assert_eq!(x.len(), y.len());
        let n = x.len() as f64;
        if n < 2.0 { return; }
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xy: f64 = x.iter().zip(y.iter()).map(|(&xi, &yi)| xi * yi).sum();
        let sum_xx: f64 = x.iter().map(|&xi| xi * xi).sum();
        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-15 { return; }
        self.slope = (n * sum_xy - sum_x * sum_y) / denom;
        self.intercept = (sum_y - self.slope * sum_x) / n;
        let mean_y = sum_y / n;
        let ss_tot: f64 = y.iter().map(|&yi| (yi - mean_y).powi(2)).sum();
        let ss_res: f64 = x.iter().zip(y.iter()).map(|(&xi, &yi)| (yi - self.predict(xi)).powi(2)).sum();
        self.r_squared = if ss_tot.abs() < 1e-15 { 1.0 } else { 1.0 - ss_res / ss_tot };
        self.fitted = true;
    }
    pub fn predict(&self, x: f64) -> f64 { self.slope * x + self.intercept }
    pub fn predict_many(&self, xs: &[f64]) -> Vec<f64> { xs.iter().map(|&x| self.predict(x)).collect() }
    pub fn residuals(&self, x: &[f64], y: &[f64]) -> Vec<f64> {
        x.iter().zip(y.iter()).map(|(&xi, &yi)| yi - self.predict(xi)).collect()
    }
    pub fn mse(&self, x: &[f64], y: &[f64]) -> f64 {
        let res = self.residuals(x, y);
        res.iter().map(|r| r * r).sum::<f64>() / res.len() as f64
    }
    pub fn rmse(&self, x: &[f64], y: &[f64]) -> f64 { self.mse(x, y).sqrt() }
}

impl fmt::Display for MatReprLinearRegression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "y = {:.4}x + {:.4} (R2={:.4})", self.slope, self.intercept, self.r_squared)
    }
}

/// Weighted undirected graph for MatRepr.
#[derive(Debug, Clone)]
pub struct MatReprWeightedGraph {
    pub adj: Vec<Vec<(usize, f64)>>,
    pub num_nodes: usize,
    pub num_edges: usize,
}

impl MatReprWeightedGraph {
    pub fn new(n: usize) -> Self { MatReprWeightedGraph { adj: vec![Vec::new(); n], num_nodes: n, num_edges: 0 } }
    pub fn add_edge(&mut self, u: usize, v: usize, w: f64) {
        self.adj[u].push((v, w));
        self.adj[v].push((u, w));
        self.num_edges += 1;
    }
    pub fn neighbors(&self, u: usize) -> &[(usize, f64)] { &self.adj[u] }
    pub fn degree(&self, u: usize) -> usize { self.adj[u].len() }
    pub fn total_weight(&self) -> f64 {
        self.adj.iter().flat_map(|edges| edges.iter().map(|(_, w)| w)).sum::<f64>() / 2.0
    }
    pub fn min_spanning_tree_weight(&self) -> f64 {
        // Kruskal's algorithm
        let mut edges: Vec<(f64, usize, usize)> = Vec::new();
        for u in 0..self.num_nodes {
            for &(v, w) in &self.adj[u] {
                if u < v { edges.push((w, u, v)); }
            }
        }
        edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let mut parent: Vec<usize> = (0..self.num_nodes).collect();
        let mut rank = vec![0usize; self.num_nodes];
        fn find_matrepr(parent: &mut Vec<usize>, x: usize) -> usize {
            if parent[x] != x { parent[x] = find_matrepr(parent, parent[x]); }
            parent[x]
        }
        let mut total = 0.0;
        let mut count = 0;
        for (w, u, v) in edges {
            let ru = find_matrepr(&mut parent, u);
            let rv = find_matrepr(&mut parent, v);
            if ru != rv {
                if rank[ru] < rank[rv] { parent[ru] = rv; }
                else if rank[ru] > rank[rv] { parent[rv] = ru; }
                else { parent[rv] = ru; rank[ru] += 1; }
                total += w;
                count += 1;
                if count == self.num_nodes - 1 { break; }
            }
        }
        total
    }
    pub fn dijkstra(&self, start: usize) -> Vec<f64> {
        let mut dist = vec![f64::INFINITY; self.num_nodes];
        let mut visited = vec![false; self.num_nodes];
        dist[start] = 0.0;
        for _ in 0..self.num_nodes {
            let mut u = None;
            let mut min_d = f64::INFINITY;
            for v in 0..self.num_nodes { if !visited[v] && dist[v] < min_d { min_d = dist[v]; u = Some(v); } }
            let u = match u { Some(v) => v, None => break };
            visited[u] = true;
            for &(v, w) in &self.adj[u] {
                let alt = dist[u] + w;
                if alt < dist[v] { dist[v] = alt; }
            }
        }
        dist
    }
    pub fn eccentricity(&self, u: usize) -> f64 {
        let dists = self.dijkstra(u);
        dists.iter().cloned().filter(|&d| d.is_finite()).fold(0.0f64, f64::max)
    }
    pub fn diameter(&self) -> f64 {
        (0..self.num_nodes).map(|u| self.eccentricity(u)).fold(0.0f64, f64::max)
    }
    pub fn clustering_coefficient(&self, u: usize) -> f64 {
        let neighbors: Vec<usize> = self.adj[u].iter().map(|(v, _)| *v).collect();
        let k = neighbors.len();
        if k < 2 { return 0.0; }
        let mut triangles = 0;
        for i in 0..k {
            for j in (i+1)..k {
                if self.adj[neighbors[i]].iter().any(|(v, _)| *v == neighbors[j]) {
                    triangles += 1;
                }
            }
        }
        2.0 * triangles as f64 / (k * (k - 1)) as f64
    }
    pub fn average_clustering_coefficient(&self) -> f64 {
        let sum: f64 = (0..self.num_nodes).map(|u| self.clustering_coefficient(u)).sum();
        sum / self.num_nodes as f64
    }
}

impl fmt::Display for MatReprWeightedGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "WGraph(n={}, e={})", self.num_nodes, self.num_edges)
    }
}

/// Moving average for MatRepr.
pub fn matrepr_moving_average(data: &[f64], window: usize) -> Vec<f64> {
    if data.len() < window { return Vec::new(); }
    let mut result = Vec::with_capacity(data.len() - window + 1);
    let mut sum: f64 = data[..window].iter().sum();
    result.push(sum / window as f64);
    for i in window..data.len() {
        sum += data[i] - data[i - window];
        result.push(sum / window as f64);
    }
    result
}

/// Cumulative sum for MatRepr.
pub fn matrepr_cumsum(data: &[f64]) -> Vec<f64> {
    let mut result = Vec::with_capacity(data.len());
    let mut sum = 0.0;
    for &v in data { sum += v; result.push(sum); }
    result
}

/// Numerical differentiation for MatRepr.
pub fn matrepr_diff(data: &[f64]) -> Vec<f64> {
    if data.len() < 2 { return Vec::new(); }
    data.windows(2).map(|w| w[1] - w[0]).collect()
}

/// Auto-correlation for MatRepr.
pub fn matrepr_autocorrelation(data: &[f64], lag: usize) -> f64 {
    if data.len() <= lag { return 0.0; }
    let n = data.len();
    let mean: f64 = data.iter().sum::<f64>() / n as f64;
    let var: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    if var.abs() < 1e-15 { return 0.0; }
    let cov: f64 = (0..(n - lag)).map(|i| (data[i] - mean) * (data[i + lag] - mean)).sum::<f64>() / n as f64;
    cov / var
}

/// Discrete Fourier transform magnitude for MatRepr.
pub fn matrepr_dft_magnitude(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    let mut magnitudes = Vec::with_capacity(n / 2 + 1);
    for k in 0..=n/2 {
        let mut re = 0.0;
        let mut im = 0.0;
        for (j, &x) in data.iter().enumerate() {
            let angle = -2.0 * std::f64::consts::PI * k as f64 * j as f64 / n as f64;
            re += x * angle.cos();
            im += x * angle.sin();
        }
        magnitudes.push((re * re + im * im).sqrt());
    }
    magnitudes
}

/// Trapezoidal integration for MatRepr.
pub fn matrepr_integrate_trapezoid(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let mut total = 0.0;
    for i in 1..x.len() {
        total += (x[i] - x[i-1]) * (y[i] + y[i-1]) / 2.0;
    }
    total
}

/// Simpson's rule integration for MatRepr.
pub fn matrepr_integrate_simpson(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let n = x.len();
    if n < 3 || n % 2 == 0 { return 0.0; }
    let mut total = 0.0;
    for i in (0..n-2).step_by(2) {
        let h = (x[i+2] - x[i]) / 6.0;
        total += h * (y[i] + 4.0 * y[i+1] + y[i+2]);
    }
    total
}

/// Convolution for MatRepr.
pub fn matrepr_convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
    let n = a.len() + b.len() - 1;
    let mut result = vec![0.0; n];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            result[i + j] += ai * bj;
        }
    }
    result
}

/// Histogram for MatRepr data analysis.
#[derive(Debug, Clone)]
pub struct MatReprHistogramExt {
    pub bins: Vec<usize>,
    pub edges: Vec<f64>,
    pub total: usize,
}

impl MatReprHistogramExt {
    pub fn from_data(data: &[f64], num_bins: usize) -> Self {
        let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = if (max - min).abs() < 1e-15 { 1.0 } else { max - min };
        let bin_width = range / num_bins as f64;
        let mut edges = Vec::with_capacity(num_bins + 1);
        for i in 0..=num_bins { edges.push(min + i as f64 * bin_width); }
        let mut bins = vec![0usize; num_bins];
        for &v in data {
            let idx = ((v - min) / bin_width).floor() as usize;
            let idx = idx.min(num_bins - 1);
            bins[idx] += 1;
        }
        MatReprHistogramExt { bins, edges, total: data.len() }
    }
    pub fn bin_count(&self, i: usize) -> usize { self.bins[i] }
    pub fn bin_density(&self, i: usize) -> f64 {
        let w = self.edges[i + 1] - self.edges[i];
        if w.abs() < 1e-15 || self.total == 0 { 0.0 }
        else { self.bins[i] as f64 / (self.total as f64 * w) }
    }
    pub fn num_bins(&self) -> usize { self.bins.len() }
    pub fn mode_bin(&self) -> usize {
        self.bins.iter().enumerate().max_by_key(|(_, &c)| c).map(|(i, _)| i).unwrap_or(0)
    }
    pub fn cumulative(&self) -> Vec<usize> {
        let mut cum = Vec::with_capacity(self.bins.len());
        let mut sum = 0;
        for &c in &self.bins { sum += c; cum.push(sum); }
        cum
    }
    pub fn percentile_bin(&self, p: f64) -> usize {
        let target = (p * self.total as f64).ceil() as usize;
        let cum = self.cumulative();
        cum.iter().position(|&c| c >= target).unwrap_or(self.bins.len() - 1)
    }
    pub fn entropy(&self) -> f64 {
        let n = self.total as f64;
        if n < 1.0 { return 0.0; }
        self.bins.iter().filter(|&&c| c > 0).map(|&c| {
            let p = c as f64 / n;
            -p * p.ln()
        }).sum()
    }
}

impl fmt::Display for MatReprHistogramExt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Hist(bins={}, total={})", self.num_bins(), self.total)
    }
}

/// Axis-aligned bounding box for MatRepr spatial indexing.
#[derive(Debug, Clone, Copy)]
pub struct MatReprAABB {
    pub x_min: f64, pub y_min: f64,
    pub x_max: f64, pub y_max: f64,
}

impl MatReprAABB {
    pub fn new(x_min: f64, y_min: f64, x_max: f64, y_max: f64) -> Self { MatReprAABB { x_min, y_min, x_max, y_max } }
    pub fn contains(&self, x: f64, y: f64) -> bool { x >= self.x_min && x <= self.x_max && y >= self.y_min && y <= self.y_max }
    pub fn intersects(&self, other: &Self) -> bool {
        !(self.x_max < other.x_min || self.x_min > other.x_max || self.y_max < other.y_min || self.y_min > other.y_max)
    }
    pub fn width(&self) -> f64 { self.x_max - self.x_min }
    pub fn height(&self) -> f64 { self.y_max - self.y_min }
    pub fn area(&self) -> f64 { self.width() * self.height() }
    pub fn center(&self) -> (f64, f64) { ((self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0) }
    pub fn subdivide(&self) -> [Self; 4] {
        let (cx, cy) = self.center();
        [
            MatReprAABB::new(self.x_min, self.y_min, cx, cy),
            MatReprAABB::new(cx, self.y_min, self.x_max, cy),
            MatReprAABB::new(self.x_min, cy, cx, self.y_max),
            MatReprAABB::new(cx, cy, self.x_max, self.y_max),
        ]
    }
}

/// 2D point for MatRepr.
#[derive(Debug, Clone, Copy)]
pub struct MatReprPoint2D { pub x: f64, pub y: f64, pub data: f64 }

/// Quadtree for MatRepr spatial indexing.
#[derive(Debug, Clone)]
pub struct MatReprQuadTree {
    pub boundary: MatReprAABB,
    pub points: Vec<MatReprPoint2D>,
    pub children: Option<Vec<MatReprQuadTree>>,
    pub capacity: usize,
    pub depth: usize,
    pub max_depth: usize,
}

impl MatReprQuadTree {
    pub fn new(boundary: MatReprAABB, capacity: usize, max_depth: usize) -> Self {
        MatReprQuadTree { boundary, points: Vec::new(), children: None, capacity, depth: 0, max_depth }
    }
    fn with_depth(boundary: MatReprAABB, capacity: usize, depth: usize, max_depth: usize) -> Self {
        MatReprQuadTree { boundary, points: Vec::new(), children: None, capacity, depth, max_depth }
    }
    pub fn insert(&mut self, p: MatReprPoint2D) -> bool {
        if !self.boundary.contains(p.x, p.y) { return false; }
        if self.points.len() < self.capacity && self.children.is_none() {
            self.points.push(p); return true;
        }
        if self.children.is_none() && self.depth < self.max_depth { self.subdivide_tree(); }
        if let Some(ref mut children) = self.children {
            for child in children.iter_mut() { if child.insert(p) { return true; } }
        }
        self.points.push(p); true
    }
    fn subdivide_tree(&mut self) {
        let quads = self.boundary.subdivide();
        let mut children = Vec::with_capacity(4);
        for q in quads.iter() {
            children.push(MatReprQuadTree::with_depth(*q, self.capacity, self.depth + 1, self.max_depth));
        }
        let old_points: Vec<_> = self.points.drain(..).collect();
        self.children = Some(children);
        for p in old_points { self.insert(p); }
    }
    pub fn query_range(&self, range: &MatReprAABB) -> Vec<MatReprPoint2D> {
        let mut result = Vec::new();
        if !self.boundary.intersects(range) { return result; }
        for p in &self.points { if range.contains(p.x, p.y) { result.push(*p); } }
        if let Some(ref children) = self.children {
            for child in children { result.extend(child.query_range(range)); }
        }
        result
    }
    pub fn count(&self) -> usize {
        let mut c = self.points.len();
        if let Some(ref children) = self.children {
            for child in children { c += child.count(); }
        }
        c
    }
    pub fn tree_depth(&self) -> usize {
        if let Some(ref children) = self.children {
            1 + children.iter().map(|c| c.tree_depth()).max().unwrap_or(0)
        } else { 0 }
    }
}

impl fmt::Display for MatReprQuadTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "QTree(count={}, depth={})", self.count(), self.tree_depth())
    }
}

/// QR decomposition helper for MatRepr.
pub fn matrepr_qr_decompose(a: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let m = a.len();
    if m == 0 { return (Vec::new(), Vec::new()); }
    let n = a[0].len();
    let mut q = vec![vec![0.0; m]; n]; // column vectors
    let mut r = vec![vec![0.0; n]; n];
    // extract columns of a
    let mut cols: Vec<Vec<f64>> = (0..n).map(|j| (0..m).map(|i| a[i][j]).collect()).collect();
    for j in 0..n {
        let mut v = cols[j].clone();
        for i in 0..j {
            let dot: f64 = v.iter().zip(q[i].iter()).map(|(&a, &b)| a * b).sum();
            r[i][j] = dot;
            for k in 0..m { v[k] -= dot * q[i][k]; }
        }
        let norm: f64 = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
        r[j][j] = norm;
        if norm.abs() > 1e-15 { for k in 0..m { q[j][k] = v[k] / norm; } }
    }
    // convert q from list of column vectors to matrix
    let q_mat: Vec<Vec<f64>> = (0..m).map(|i| (0..n).map(|j| q[j][i]).collect()).collect();
    (q_mat, r)
}

/// Solve upper triangular system Rx = b for MatRepr.
pub fn matrepr_solve_upper_triangular(r: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i+1)..n { s -= r[i][j] * x[j]; }
        x[i] = if r[i][i].abs() > 1e-15 { s / r[i][i] } else { 0.0 };
    }
    x
}

/// Matrix-vector multiply for MatRepr.
pub fn matrepr_mat_vec_mul(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter().map(|row| row.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum()).collect()
}

/// Matrix transpose for MatRepr.
pub fn matrepr_transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if a.is_empty() { return Vec::new(); }
    let m = a.len(); let n = a[0].len();
    (0..n).map(|j| (0..m).map(|i| a[i][j]).collect()).collect()
}

/// Matrix multiply for MatRepr.
pub fn matrepr_mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    if m == 0 { return Vec::new(); }
    let k = a[0].len();
    let n = b[0].len();
    let mut c = vec![vec![0.0; n]; m];
    for i in 0..m { for j in 0..n { for l in 0..k { c[i][j] += a[i][l] * b[l][j]; } } }
    c
}

/// Frobenius norm for MatRepr.
pub fn matrepr_frobenius_norm(a: &[Vec<f64>]) -> f64 {
    a.iter().flat_map(|row| row.iter()).map(|&x| x * x).sum::<f64>().sqrt()
}

/// Matrix trace for MatRepr.
pub fn matrepr_trace(a: &[Vec<f64>]) -> f64 {
    a.iter().enumerate().map(|(i, row)| if i < row.len() { row[i] } else { 0.0 }).sum()
}

/// Identity matrix for MatRepr.
pub fn matrepr_identity(n: usize) -> Vec<Vec<f64>> {
    let mut m = vec![vec![0.0; n]; n];
    for i in 0..n { m[i][i] = 1.0; }
    m
}

/// Power iteration for dominant eigenvalue for MatRepr.
pub fn matrepr_power_iteration(a: &[Vec<f64>], max_iter: usize) -> (f64, Vec<f64>) {
    let n = a.len();
    let mut v = vec![1.0; n];
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    for x in &mut v { *x /= norm; }
    let mut eigenvalue = 0.0;
    for _ in 0..max_iter {
        let av = matrepr_mat_vec_mul(a, &v);
        let norm: f64 = av.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm.abs() < 1e-15 { break; }
        eigenvalue = av.iter().zip(v.iter()).map(|(&a, &b)| a * b).sum();
        v = av.iter().map(|&x| x / norm).collect();
    }
    (eigenvalue, v)
}

/// Running statistics with min/max tracking for MatRepr.
#[derive(Debug, Clone)]
pub struct MatReprRunningStats {
    pub count: u64,
    pub mean: f64,
    pub m2: f64,
    pub min_val: f64,
    pub max_val: f64,
    pub sum: f64,
}

impl MatReprRunningStats {
    pub fn new() -> Self { MatReprRunningStats { count: 0, mean: 0.0, m2: 0.0, min_val: f64::INFINITY, max_val: f64::NEG_INFINITY, sum: 0.0 } }
    pub fn push(&mut self, x: f64) {
        self.count += 1;
        let delta = x - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
        self.min_val = self.min_val.min(x);
        self.max_val = self.max_val.max(x);
        self.sum += x;
    }
    pub fn variance(&self) -> f64 { if self.count < 2 { 0.0 } else { self.m2 / (self.count - 1) as f64 } }
    pub fn std_dev(&self) -> f64 { self.variance().sqrt() }
    pub fn range(&self) -> f64 { self.max_val - self.min_val }
    pub fn coefficient_of_variation(&self) -> f64 { if self.mean.abs() < 1e-15 { 0.0 } else { self.std_dev() / self.mean.abs() } }
    pub fn merge(&mut self, other: &Self) {
        if other.count == 0 { return; }
        let combined_count = self.count + other.count;
        let delta = other.mean - self.mean;
        let combined_mean = self.mean + delta * other.count as f64 / combined_count as f64;
        self.m2 += other.m2 + delta * delta * self.count as f64 * other.count as f64 / combined_count as f64;
        self.mean = combined_mean;
        self.count = combined_count;
        self.min_val = self.min_val.min(other.min_val);
        self.max_val = self.max_val.max(other.max_val);
        self.sum += other.sum;
    }
}

impl fmt::Display for MatReprRunningStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Stats(n={}, mean={:.4}, std={:.4})", self.count, self.mean, self.std_dev())
    }
}

/// Percentile calculator for MatRepr.
pub fn matrepr_percentile_at(data: &[f64], p: f64) -> f64 {
    if data.is_empty() { return 0.0; }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = p / 100.0 * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi { sorted[lo] }
    else { sorted[lo] * (hi as f64 - idx) + sorted[hi] * (idx - lo as f64) }
}

/// Interquartile range for MatRepr.
pub fn matrepr_iqr(data: &[f64]) -> f64 {
    matrepr_percentile_at(data, 75.0) - matrepr_percentile_at(data, 25.0)
}

/// Detect outliers using IQR method for MatRepr.
pub fn matrepr_outliers(data: &[f64]) -> Vec<usize> {
    let q1 = matrepr_percentile_at(data, 25.0);
    let q3 = matrepr_percentile_at(data, 75.0);
    let iqr = q3 - q1;
    let lower = q1 - 1.5 * iqr;
    let upper = q3 + 1.5 * iqr;
    data.iter().enumerate().filter(|(_, &v)| v < lower || v > upper).map(|(i, _)| i).collect()
}

/// Z-score normalization for MatRepr.
pub fn matrepr_zscore(data: &[f64]) -> Vec<f64> {
    let n = data.len() as f64;
    if n < 2.0 { return data.to_vec(); }
    let mean = data.iter().sum::<f64>() / n;
    let std = (data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    if std.abs() < 1e-15 { return vec![0.0; data.len()]; }
    data.iter().map(|&x| (x - mean) / std).collect()
}

/// Rank values for MatRepr.
pub fn matrepr_rank(data: &[f64]) -> Vec<f64> {
    let mut indexed: Vec<(usize, f64)> = data.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let mut ranks = vec![0.0; data.len()];
    for (rank, &(idx, _)) in indexed.iter().enumerate() { ranks[idx] = (rank + 1) as f64; }
    ranks
}

/// Spearman rank correlation for MatRepr.
pub fn matrepr_spearman(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let rx = matrepr_rank(x);
    let ry = matrepr_rank(y);
    let n = x.len() as f64;
    let d_sq: f64 = rx.iter().zip(ry.iter()).map(|(&a, &b)| (a - b).powi(2)).sum();
    1.0 - 6.0 * d_sq / (n * (n * n - 1.0))
}

/// Geometric mean for MatRepr.
pub fn matrepr_geomean(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    let log_sum: f64 = data.iter().map(|&x| x.ln()).sum();
    (log_sum / data.len() as f64).exp()
}

/// Harmonic mean for MatRepr.
pub fn matrepr_harmmean(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    let recip_sum: f64 = data.iter().map(|&x| 1.0 / x).sum();
    data.len() as f64 / recip_sum
}

/// Skewness for MatRepr.
pub fn matrepr_sample_skewness(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    if n < 3.0 { return 0.0; }
    let mean = data.iter().sum::<f64>() / n;
    let m2: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum();
    let m3: f64 = data.iter().map(|&x| (x - mean).powi(3)).sum();
    let s2 = m2 / (n - 1.0);
    let s = s2.sqrt();
    if s.abs() < 1e-15 { return 0.0; }
    (n / ((n - 1.0) * (n - 2.0))) * m3 / s.powi(3)
}

/// Excess kurtosis for MatRepr.
pub fn matrepr_excess_kurtosis(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    if n < 4.0 { return 0.0; }
    let mean = data.iter().sum::<f64>() / n;
    let m2: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    let m4: f64 = data.iter().map(|&x| (x - mean).powi(4)).sum::<f64>() / n;
    if m2.abs() < 1e-15 { return 0.0; }
    m4 / (m2 * m2) - 3.0
}

/// Covariance matrix for MatRepr.
pub fn matrepr_covariance_matrix(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if data.is_empty() { return Vec::new(); }
    let n = data.len() as f64;
    let d = data[0].len();
    let means: Vec<f64> = (0..d).map(|j| data.iter().map(|row| row[j]).sum::<f64>() / n).collect();
    let mut cov = vec![vec![0.0; d]; d];
    for i in 0..d {
        for j in i..d {
            let c: f64 = data.iter().map(|row| (row[i] - means[i]) * (row[j] - means[j])).sum::<f64>() / (n - 1.0).max(1.0);
            cov[i][j] = c; cov[j][i] = c;
        }
    }
    cov
}

/// Correlation matrix for MatRepr.
pub fn matrepr_correlation_matrix(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let cov = matrepr_covariance_matrix(data);
    let d = cov.len();
    let mut corr = vec![vec![0.0; d]; d];
    for i in 0..d {
        for j in 0..d {
            let denom = (cov[i][i] * cov[j][j]).sqrt();
            corr[i][j] = if denom.abs() < 1e-15 { 0.0 } else { cov[i][j] / denom };
        }
    }
    corr
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_arithmetic() {
        let a = Complex::new(3.0, 4.0);
        let b = Complex::new(1.0, -2.0);
        let sum = a + b;
        assert!((sum.re - 4.0).abs() < 1e-12);
        assert!((sum.im - 2.0).abs() < 1e-12);
        let prod = a * b;
        assert!((prod.re - 11.0).abs() < 1e-12);
        assert!((prod.im - (-2.0)).abs() < 1e-12);
    }

    #[test]
    fn test_complex_norm() {
        let z = Complex::new(3.0, 4.0);
        assert!((z.norm() - 5.0).abs() < 1e-12);
        assert!((z.conjugate().im - (-4.0)).abs() < 1e-12);
    }

    #[test]
    fn test_complex_division() {
        let a = Complex::new(1.0, 0.0);
        let b = Complex::new(0.0, 1.0);
        let q = a / b;
        assert!((q.re - 0.0).abs() < 1e-12);
        assert!((q.im - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn test_matrix_identity() {
        let id = Matrix::identity(3);
        assert!((id.trace() - 3.0).abs() < 1e-12);
        assert!((id.determinant() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_matrix_multiply() {
        let a = Matrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let b = Matrix::from_rows(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
        let c = a.multiply(&b);
        assert!((c.data[0][0] - 19.0).abs() < 1e-12);
        assert!((c.data[0][1] - 22.0).abs() < 1e-12);
        assert!((c.data[1][0] - 43.0).abs() < 1e-12);
        assert!((c.data[1][1] - 50.0).abs() < 1e-12);
    }

    #[test]
    fn test_matrix_determinant() {
        let m = Matrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        assert!((m.determinant() - (-2.0)).abs() < 1e-12);
    }

    #[test]
    fn test_matrix_inverse() {
        let m = Matrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let inv = m.inverse().unwrap();
        let product = m.multiply(&inv);
        let id = Matrix::identity(2);
        assert!(product.sub(&id).frobenius_norm() < 1e-10);
    }

    #[test]
    fn test_matrix_singular_no_inverse() {
        let m = Matrix::from_rows(vec![vec![1.0, 2.0], vec![2.0, 4.0]]);
        assert!(m.inverse().is_none());
    }

    #[test]
    fn test_matrix_transpose() {
        let m = Matrix::from_rows(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        let t = m.transpose();
        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 2);
        assert!((t.data[0][1] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_matrix_rank() {
        let m = Matrix::from_rows(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0], vec![7.0, 8.0, 9.0]]);
        assert_eq!(m.rank(), 2);
        let id = Matrix::identity(3);
        assert_eq!(id.rank(), 3);
    }

    #[test]
    fn test_matrix_kronecker() {
        let a = Matrix::identity(2);
        let b = Matrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let k = a.kronecker(&b);
        assert_eq!(k.rows, 4);
        assert_eq!(k.cols, 4);
        assert!((k.data[0][0] - 1.0).abs() < 1e-12);
        assert!((k.data[2][2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_matrix_direct_sum() {
        let a = Matrix::identity(2);
        let b = Matrix::from_rows(vec![vec![5.0]]);
        let ds = a.direct_sum(&b);
        assert_eq!(ds.rows, 3);
        assert_eq!(ds.cols, 3);
        assert!((ds.data[2][2] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_matrix_power() {
        let m = Matrix::from_rows(vec![vec![1.0, 1.0], vec![0.0, 1.0]]);
        let m3 = m.power(3);
        assert!((m3.data[0][1] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_matrix_is_symmetric() {
        let m = Matrix::from_rows(vec![vec![1.0, 2.0], vec![2.0, 3.0]]);
        assert!(m.is_symmetric());
        let m2 = Matrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        assert!(!m2.is_symmetric());
    }

    #[test]
    fn test_polynomial_evaluate() {
        let p = Polynomial::new(vec![1.0, 2.0, 3.0]); // 1 + 2x + 3x²
        assert!((p.evaluate(2.0) - 17.0).abs() < 1e-12);
    }

    #[test]
    fn test_polynomial_multiply() {
        let p1 = Polynomial::new(vec![1.0, 1.0]); // 1 + x
        let p2 = Polynomial::new(vec![1.0, -1.0]); // 1 - x
        let prod = p1.mul(&p2); // 1 - x²
        assert!((prod.coeffs[0] - 1.0).abs() < 1e-12);
        assert!(prod.coeffs[1].abs() < 1e-12);
        assert!((prod.coeffs[2] - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn test_polynomial_derivative() {
        let p = Polynomial::new(vec![1.0, 2.0, 3.0]); // 1 + 2x + 3x²
        let dp = p.derivative(); // 2 + 6x
        assert!((dp.coeffs[0] - 2.0).abs() < 1e-12);
        assert!((dp.coeffs[1] - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_polynomial_roots_quadratic() {
        let p = Polynomial::new(vec![-6.0, 1.0, 1.0]); // x² + x - 6
        let (r1, r2) = p.roots_quadratic().unwrap();
        let roots = vec![r1.re, r2.re];
        assert!(roots.iter().any(|&r| (r - 2.0).abs() < 1e-10));
        assert!(roots.iter().any(|&r| (r - (-3.0)).abs() < 1e-10));
    }

    #[test]
    fn test_characteristic_polynomial_2x2() {
        let m = Matrix::from_rows(vec![vec![2.0, 1.0], vec![1.0, 2.0]]);
        let cp = m.characteristic_polynomial();
        // Eigenvalues: 3, 1. Char poly: λ² - 4λ + 3
        assert!((cp.evaluate(3.0)).abs() < 1e-10);
        assert!((cp.evaluate(1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_eigenvalues_2x2() {
        let m = Matrix::from_rows(vec![vec![2.0, 1.0], vec![1.0, 2.0]]);
        let (e1, e2) = m.eigenvalues_2x2().unwrap();
        let vals = vec![e1.re, e2.re];
        assert!(vals.iter().any(|&v| (v - 3.0).abs() < 1e-10));
        assert!(vals.iter().any(|&v| (v - 1.0).abs() < 1e-10));
    }

    #[test]
    fn test_permutation_to_matrix() {
        let m = permutation_to_matrix(&[1, 2, 0]);
        assert!((m.data[0][1] - 1.0).abs() < 1e-12);
        assert!((m.data[1][2] - 1.0).abs() < 1e-12);
        assert!((m.data[2][0] - 1.0).abs() < 1e-12);
        assert!((m.determinant().abs() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_representation_kernel() {
        let mut rep = Representation::new(2);
        rep.add_element(0, "e", Matrix::identity(2));
        rep.add_element(1, "a", Matrix::from_rows(vec![vec![-1.0, 0.0], vec![0.0, -1.0]]));
        assert_eq!(rep.kernel().len(), 1);
        assert!(rep.is_faithful());
    }

    #[test]
    fn test_character_from_representation() {
        let mut rep = Representation::new(2);
        rep.add_element(0, "e", Matrix::identity(2));
        rep.add_element(1, "a", Matrix::from_rows(vec![vec![-1.0, 0.0], vec![0.0, -1.0]]));
        let chi = Character::from_representation(&rep);
        assert!((chi.values[&0].re - 2.0).abs() < 1e-12);
        assert!((chi.values[&1].re - (-2.0)).abs() < 1e-12);
    }

    #[test]
    fn test_character_inner_product() {
        let mut chi1 = Character { values: HashMap::new(), dimension: 1 };
        chi1.values.insert(0, Complex::new(1.0, 0.0));
        chi1.values.insert(1, Complex::new(1.0, 0.0));
        let mut chi2 = Character { values: HashMap::new(), dimension: 1 };
        chi2.values.insert(0, Complex::new(1.0, 0.0));
        chi2.values.insert(1, Complex::new(-1.0, 0.0));
        let ip = chi1.inner_product(&chi2, 2);
        assert!(ip.re.abs() < 0.1); // orthogonal characters
    }

    #[test]
    fn test_matrix_group_z2() {
        let neg = Matrix::from_rows(vec![vec![-1.0, 0.0], vec![0.0, -1.0]]);
        let g = MatrixGroup::from_generators(vec![neg]);
        assert_eq!(g.order(), 2);
        assert!(g.is_abelian());
    }

    #[test]
    fn test_matrix_group_z3() {
        let theta = 2.0 * std::f64::consts::PI / 3.0;
        let rot = Matrix::from_rows(vec![
            vec![theta.cos(), -theta.sin()],
            vec![theta.sin(), theta.cos()],
        ]);
        let g = MatrixGroup::from_generators(vec![rot]);
        assert_eq!(g.order(), 3);
        assert!(g.is_abelian());
    }

    #[test]
    fn test_ring_element_operations() {
        let a = RingElement::new(vec![1, 0, 1]); // ρ_0 + ρ_2
        let b = RingElement::new(vec![0, 1, 0]); // ρ_1
        let sum = a.add(&b);
        assert_eq!(sum.coefficients, vec![1, 1, 1]);
        assert_eq!(sum.dimension(), 3);
    }

    #[test]
    fn test_complex_matrix_unitary() {
        let id = ComplexMatrix::identity(2);
        assert!(id.is_unitary());
    }

    #[test]
    fn test_complex_matrix_multiply() {
        let a = ComplexMatrix::identity(2);
        let b = ComplexMatrix::identity(2);
        let c = a.multiply(&b);
        assert!((c.data[0][0].re - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_molien_series() {
        let mut rep = Representation::new(1);
        rep.add_element(0, "e", Matrix::identity(1));
        let terms = molien_series_terms(&rep, 5);
        assert!((terms[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_random_orthogonal() {
        let m = random_orthogonal(3, 42);
        assert!(m.is_orthogonal());
    }

    #[test]
    fn test_matrix_display() {
        let m = Matrix::identity(2);
        let s = format!("{}", m);
        assert!(s.contains("1.0000"));
    }

    #[test]
    fn test_polynomial_display() {
        let p = Polynomial::new(vec![1.0, 2.0, 3.0]);
        let s = format!("{}", p);
        assert!(!s.is_empty());
    }

    #[test]
    fn test_frobenius_norm() {
        let m = Matrix::from_rows(vec![vec![3.0, 4.0]]);
        assert!((m.frobenius_norm() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_row_echelon() {
        let m = Matrix::from_rows(vec![
            vec![2.0, 4.0, 6.0],
            vec![1.0, 3.0, 5.0],
        ]);
        let re = m.row_echelon();
        assert!((re.data[0][0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_matrix_zero() {
        let z = Matrix::zero(2, 3);
        assert_eq!(z.rows, 2);
        assert_eq!(z.cols, 3);
        assert!((z.frobenius_norm()).abs() < 1e-12);
    }

    #[test]
    fn test_polynomial_add() {
        let p1 = Polynomial::new(vec![1.0, 2.0]);
        let p2 = Polynomial::new(vec![3.0, 4.0, 5.0]);
        let sum = p1.add(&p2);
        assert!((sum.coeffs[0] - 4.0).abs() < 1e-12);
        assert!((sum.coeffs[2] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_character_decomposition_display() {
        let cd = CharacterDecomposition {
            multiplicities: vec![1, 0, 2],
            irreducibles: vec![0, 2],
        };
        let s = format!("{}", cd);
        assert!(s.contains("ρ_0"));
        assert!(s.contains("2·ρ_2"));
    }

    #[test]
    fn test_representation_statistics_fn() {
        let mut rep = Representation::new(2);
        rep.add_element(0, "e", Matrix::identity(2));
        let s = representation_statistics(&rep);
        assert!(s.contains("Dimension: 2"));
    }

    #[test]
    fn test_complex_exp() {
        let z = Complex::new(0.0, std::f64::consts::PI);
        let e = z.exp();
        assert!((e.re - (-1.0)).abs() < 1e-10); // e^(iπ) = -1
        assert!(e.im.abs() < 1e-10);
    }
    #[test]
    fn test_schurdecomposition_new() {
        let item = SchurDecomposition::new(Vec::new(), Vec::new(), 0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_qrfactorization_new() {
        let item = QrFactorization::new(Vec::new(), Vec::new(), 0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_svdresult_new() {
        let item = SvdResult::new(Vec::new(), Vec::new(), Vec::new(), 0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_matrixexponential_new() {
        let item = MatrixExponential::new(Vec::new(), Vec::new(), 0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_matrixlogarithm_new() {
        let item = MatrixLogarithm::new(Vec::new(), Vec::new(), false);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_cayleyhamilton_new() {
        let item = CayleyHamilton::new(Vec::new(), 0, false);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_minimalpolynomial_new() {
        let item = MinimalPolynomial::new(Vec::new(), 0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_jordanblock_new() {
        let item = JordanBlock::new(0.0, 0, 0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_jordanform_new() {
        let item = JordanForm::new(Vec::new(), Vec::new(), 0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_orthogonalprojection_new() {
        let item = OrthogonalProjection::new(Vec::new(), Vec::new(), 0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_gramschmidt_new() {
        let item = GramSchmidt::new(Vec::new(), Vec::new(), 0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_poweriteration_new() {
        let item = PowerIteration::new(0.0, Vec::new(), 0, false);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_matrix_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let m = matrix_mean(&data);
        assert!((m - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_variance() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let v = matrix_variance(&data);
        assert!(v > 0.0);
    }

    #[test]
    fn test_matrix_median() {
        let data = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        let m = matrix_median(&data);
        assert!((m - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_entropy() {
        let data = vec![1.0, 1.0, 1.0, 1.0];
        let e = matrix_entropy(&data);
        assert!(e > 0.0);
    }

    #[test]
    fn test_matrix_std_dev() {
        let data = vec![10.0, 10.0, 10.0];
        let s = matrix_std_dev(&data);
        assert!(s.abs() < 1e-10);
    }

    #[test]
    fn test_matrix_analysis() {
        let mut a = MatrixAnalysis::new(3);
        a.set(0, 1, 0.5);
        a.set(1, 2, 0.3);
        assert!((a.get(0, 1) - 0.5).abs() < 1e-10);
        assert_eq!(a.size, 3);
    }

    #[test]
    fn test_matrix_iterator() {
        let iter = MatrixResultIterator::new(vec![(0, 1.0), (1, 2.0), (2, 3.0)]);
        let items: Vec<_> = iter.collect();
        assert_eq!(items.len(), 3);
    }

    #[test]
    fn test_matrix_batch_processor() {
        let mut proc = MatrixBatchProcessor::new(2);
        proc.process_batch(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(proc.processed, 4);
        assert_eq!(proc.results.len(), 2);
    }

    #[test]
    fn test_matrix_histogram() {
        let hist = MatrixHistogram::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0], 3);
        assert_eq!(hist.num_bins(), 3);
        assert_eq!(hist.total_count, 5);
    }

    #[test]
    fn test_matrix_graph() {
        let mut g = MatrixGraph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 2.0);
        g.add_edge(2, 3, 3.0);
        assert_eq!(g.edge_count, 3);
        assert!(g.has_edge(0, 1));
        assert!(!g.has_edge(3, 0));
        assert!(g.is_acyclic());
    }

    #[test]
    fn test_matrix_graph_shortest_path() {
        let mut g = MatrixGraph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 2.0);
        g.add_edge(0, 2, 10.0);
        let dist = g.shortest_path_dijkstra(0);
        assert!((dist[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_graph_topo_sort() {
        let mut g = MatrixGraph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        let topo = g.topological_sort();
        assert!(topo.is_some());
    }

    #[test]
    fn test_matrix_graph_components() {
        let mut g = MatrixGraph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(2, 3, 1.0);
        let comps = g.connected_components();
        assert_eq!(comps.len(), 2);
    }

    #[test]
    fn test_matrix_cache() {
        let mut cache = MatrixCache::new(10);
        cache.insert(42, vec![1.0, 2.0]);
        assert!(cache.get(42).is_some());
        assert!(cache.get(99).is_none());
    }

    #[test]
    fn test_matrix_config() {
        let config = MatrixConfig::default_config().with_verbose(true).with_max_iterations(500);
        assert!(config.verbose);
        assert_eq!(config.max_iterations, 500);
    }

    #[test]
    fn test_matrix_report() {
        let mut report = MatrixReport::new("Test Report");
        report.add_metric("accuracy", 0.95);
        report.add_warning("low sample size");
        assert_eq!(report.total_metrics(), 1);
        assert!(report.has_warnings());
        let text = report.render_text();
        assert!(text.contains("Test Report"));
    }

    #[test]
    fn test_matrix_kmeans() {
        let data = vec![vec![0.0, 0.0], vec![0.1, 0.1], vec![10.0, 10.0], vec![10.1, 10.1]];
        let assignments = matrix_kmeans(&data, 2, 100);
        assert_eq!(assignments.len(), 4);
        assert_eq!(assignments[0], assignments[1]);
        assert_eq!(assignments[2], assignments[3]);
    }

    #[test]
    fn test_matrix_pairwise_distances() {
        let points = vec![vec![0.0, 0.0], vec![3.0, 4.0]];
        let dists = matrix_pairwise_distances(&points);
        assert!((dists[0][1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_harmonic_mean() {
        let data = vec![1.0, 2.0, 4.0];
        let hm = matrix_harmonic_mean(&data);
        assert!(hm > 0.0 && hm < 4.0);
    }

    #[test]
    fn test_matrix_geometric_mean() {
        let data = vec![1.0, 2.0, 4.0];
        let gm = matrix_geometric_mean(&data);
        assert!(gm > 0.0 && gm < 4.0);
    }

    #[test]
    fn test_matrix_skewness() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let s = matrix_skewness(&data);
        assert!(s.abs() < 1.0);
    }

    #[test]
    fn test_matrix_kurtosis() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let k = matrix_kurtosis(&data);
        let _ = k; // Just verify it computes
    }

    #[test]
    fn test_matrix_gini() {
        let data = vec![1.0, 1.0, 1.0, 1.0];
        let g = matrix_gini(&data);
        assert!(g.abs() < 0.01);
    }

    #[test]
    fn test_matrix_percentile() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let p = matrix_percentile(&data);
        assert!(p >= 9.0);
    }

    #[test]
    fn test_matrix_pca_2d() {
        let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let proj = matrix_pca_2d(&data);
        assert_eq!(proj.len(), 2);
    }

    #[test]
    fn test_matrix_analysis_normalize() {
        let mut a = MatrixAnalysis::new(2);
        a.set(0, 0, 1.0); a.set(0, 1, 3.0);
        a.normalize();
        assert!((a.total_sum() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_analysis_transpose() {
        let mut a = MatrixAnalysis::new(2);
        a.set(0, 1, 5.0);
        let t = a.transpose();
        assert!((t.get(1, 0) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_analysis_multiply() {
        let mut a = MatrixAnalysis::new(2);
        a.set(0, 0, 1.0); a.set(1, 1, 1.0);
        let mut b = MatrixAnalysis::new(2);
        b.set(0, 1, 2.0); b.set(1, 0, 3.0);
        let c = a.multiply(&b);
        assert!((c.get(0, 1) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_analysis_frobenius() {
        let mut a = MatrixAnalysis::new(2);
        a.set(0, 0, 3.0); a.set(1, 1, 4.0);
        assert!((a.frobenius_norm() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_analysis_symmetric() {
        let mut a = MatrixAnalysis::new(2);
        a.set(0, 1, 1.0); a.set(1, 0, 1.0);
        assert!(a.is_symmetric());
    }

    #[test]
    fn test_matrix_graph_dot() {
        let mut g = MatrixGraph::new(2);
        g.add_edge(0, 1, 1.0);
        let dot = g.to_dot();
        assert!(dot.contains("digraph"));
    }

    #[test]
    fn test_matrix_histogram_render() {
        let hist = MatrixHistogram::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 5);
        let ascii = hist.render_ascii(20);
        assert!(!ascii.is_empty());
    }

    #[test]
    fn test_matrix_batch_reset() {
        let mut proc = MatrixBatchProcessor::new(3);
        proc.process_batch(&[1.0, 2.0, 3.0]);
        assert!(proc.processed > 0);
        proc.reset();
        assert_eq!(proc.processed, 0);
    }

    #[test]
    fn test_matrix_graph_remove_edge() {
        let mut g = MatrixGraph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        assert_eq!(g.edge_count, 2);
        g.remove_edge(0, 1);
        assert_eq!(g.edge_count, 1);
        assert!(!g.has_edge(0, 1));
    }

    #[test]
    fn test_matrepr_dense_matrix_new() {
        let m = MatReprDenseMatrix::new(3, 3);
        assert_eq!(m.rows, 3);
        assert_eq!(m.cols, 3);
    }

    #[test]
    fn test_matrepr_dense_matrix_identity() {
        let m = MatReprDenseMatrix::identity(3);
        assert!((m.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((m.get(0, 1)).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_dense_matrix_mul() {
        let a = MatReprDenseMatrix::identity(2);
        let b = MatReprDenseMatrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let c = a.mul_matrix(&b);
        assert!((c.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((c.get(1, 1) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_dense_matrix_transpose() {
        let a = MatReprDenseMatrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t = a.transpose();
        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 2);
        assert!((t.get(0, 1) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_dense_matrix_det_2x2() {
        let m = MatReprDenseMatrix::from_vec(2, 2, vec![3.0, 7.0, 1.0, -4.0]);
        let det = m.determinant_2x2();
        assert!((det - (-19.0)).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_dense_matrix_det_3x3() {
        let m = MatReprDenseMatrix::from_vec(3, 3, vec![1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0]);
        let det = m.determinant_3x3();
        assert!((det - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_dense_matrix_inverse_2x2() {
        let m = MatReprDenseMatrix::from_vec(2, 2, vec![4.0, 7.0, 2.0, 6.0]);
        let inv = m.inverse_2x2().unwrap();
        let prod = m.mul_matrix(&inv);
        assert!((prod.get(0, 0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_dense_matrix_power() {
        let m = MatReprDenseMatrix::identity(3);
        let p = m.power(5);
        assert!((p.get(0, 0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_dense_matrix_rank() {
        let m = MatReprDenseMatrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0]);
        assert_eq!(m.rank(), 1);
    }

    #[test]
    fn test_matrepr_dense_matrix_solve() {
        let a = MatReprDenseMatrix::from_vec(2, 2, vec![2.0, 1.0, 5.0, 3.0]);
        let x = a.solve(&[4.0, 7.0]).unwrap();
        assert!((x[0] - 5.0).abs() < 1e-8);
        assert!((x[1] - (-6.0)).abs() < 1e-8);
    }

    #[test]
    fn test_matrepr_dense_matrix_lu() {
        let a = MatReprDenseMatrix::from_vec(2, 2, vec![4.0, 3.0, 6.0, 3.0]);
        let (l, u) = a.lu_decomposition();
        let prod = l.mul_matrix(&u);
        assert!((prod.get(0, 0) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_dense_matrix_eigenvalues() {
        let m = MatReprDenseMatrix::from_vec(2, 2, vec![2.0, 1.0, 1.0, 2.0]);
        let (e1, e2) = m.eigenvalues_2x2();
        assert!((e1 - 3.0).abs() < 1e-10);
        assert!((e2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_dense_matrix_kronecker() {
        let a = MatReprDenseMatrix::identity(2);
        let b = MatReprDenseMatrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let k = a.kronecker_product(&b);
        assert_eq!(k.rows, 4);
        assert_eq!(k.cols, 4);
    }

    #[test]
    fn test_matrepr_dense_matrix_hadamard() {
        let a = MatReprDenseMatrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = MatReprDenseMatrix::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let h = a.hadamard_product(&b);
        assert!((h.get(0, 0) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_interval() {
        let a = MatReprInterval::new(1.0, 3.0);
        let b = MatReprInterval::new(2.0, 5.0);
        assert!(a.overlaps(&b));
        assert!(a.contains(2.0));
        assert!(!a.contains(4.0));
        let sum = a.add(&b);
        assert!((sum.lo - 3.0).abs() < 1e-10);
        assert!((sum.hi - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_interval_mul() {
        let a = MatReprInterval::new(-2.0, 3.0);
        let b = MatReprInterval::new(1.0, 4.0);
        let prod = a.mul(&b);
        assert!((prod.lo - (-8.0)).abs() < 1e-10);
        assert!((prod.hi - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_interval_hull() {
        let a = MatReprInterval::new(1.0, 3.0);
        let b = MatReprInterval::new(5.0, 7.0);
        let h = a.hull(&b);
        assert!((h.lo - 1.0).abs() < 1e-10);
        assert!((h.hi - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_state_machine() {
        let mut sm = MatReprStateMachine::new();
        assert_eq!(*sm.state(), MatReprState::Uninitialized);
        assert!(sm.transition(MatReprState::Loaded));
        assert_eq!(*sm.state(), MatReprState::Loaded);
        assert_eq!(sm.transition_count, 1);
    }

    #[test]
    fn test_matrepr_state_machine_invalid() {
        let mut sm = MatReprStateMachine::new();
        let last_state = MatReprState::Invalid;
        assert!(!sm.can_transition(&last_state));
    }

    #[test]
    fn test_matrepr_state_machine_reset() {
        let mut sm = MatReprStateMachine::new();
        sm.transition(MatReprState::Loaded);
        sm.reset();
        assert_eq!(*sm.state(), MatReprState::Uninitialized);
        assert_eq!(sm.history_len(), 0);
    }

    #[test]
    fn test_matrepr_ring_buffer() {
        let mut rb = MatReprRingBuffer::new(3);
        rb.push(1.0); rb.push(2.0); rb.push(3.0);
        assert!(rb.is_full());
        assert!((rb.average() - 2.0).abs() < 1e-10);
        rb.push(4.0);
        assert!((rb.oldest().unwrap() - 2.0).abs() < 1e-10);
        assert!((rb.latest().unwrap() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_ring_buffer_to_vec() {
        let mut rb = MatReprRingBuffer::new(5);
        rb.push(10.0); rb.push(20.0); rb.push(30.0);
        let v = rb.to_vec();
        assert_eq!(v.len(), 3);
        assert!((v[0] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_disjoint_set() {
        let mut ds = MatReprDisjointSet::new(5);
        assert_eq!(ds.num_components(), 5);
        ds.union(0, 1);
        ds.union(2, 3);
        assert_eq!(ds.num_components(), 3);
        assert!(ds.connected(0, 1));
        assert!(!ds.connected(0, 2));
    }

    #[test]
    fn test_matrepr_disjoint_set_components() {
        let mut ds = MatReprDisjointSet::new(4);
        ds.union(0, 1); ds.union(2, 3);
        let comps = ds.components();
        assert_eq!(comps.len(), 2);
    }

    #[test]
    fn test_matrepr_sorted_list() {
        let mut sl = MatReprSortedList::new();
        sl.insert(3.0); sl.insert(1.0); sl.insert(2.0);
        assert_eq!(sl.len(), 3);
        assert!((sl.min().unwrap() - 1.0).abs() < 1e-10);
        assert!((sl.max().unwrap() - 3.0).abs() < 1e-10);
        assert!((sl.median() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_sorted_list_remove() {
        let mut sl = MatReprSortedList::new();
        sl.insert(1.0); sl.insert(2.0); sl.insert(3.0);
        assert!(sl.remove(2.0));
        assert_eq!(sl.len(), 2);
        assert!(!sl.contains(2.0));
    }

    #[test]
    fn test_matrepr_ema() {
        let mut ema = MatReprEma::new(0.5);
        ema.update(10.0);
        assert!((ema.current() - 10.0).abs() < 1e-10);
        ema.update(20.0);
        assert!((ema.current() - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_bloom_filter() {
        let mut bf = MatReprBloomFilter::new(1000, 3);
        bf.insert(42);
        bf.insert(100);
        assert!(bf.may_contain(42));
        assert!(bf.may_contain(100));
        assert_eq!(bf.count(), 2);
    }

    #[test]
    fn test_matrepr_trie() {
        let mut trie = MatReprTrie::new();
        trie.insert("hello", 1);
        trie.insert("help", 2);
        trie.insert("world", 3);
        assert_eq!(trie.search("hello"), Some(1));
        assert_eq!(trie.search("help"), Some(2));
        assert_eq!(trie.search("hell"), None);
        assert!(trie.starts_with("hel"));
        assert!(!trie.starts_with("xyz"));
    }

    #[test]
    fn test_matrepr_dense_matrix_sym() {
        let m = MatReprDenseMatrix::from_vec(2, 2, vec![1.0, 2.0, 2.0, 3.0]);
        assert!(m.is_symmetric());
    }

    #[test]
    fn test_matrepr_dense_matrix_diag() {
        let m = MatReprDenseMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 3.0]);
        assert!(m.is_diagonal());
    }

    #[test]
    fn test_matrepr_dense_matrix_upper_tri() {
        let m = MatReprDenseMatrix::from_vec(3, 3, vec![1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 0.0, 0.0, 6.0]);
        assert!(m.is_upper_triangular());
    }

    #[test]
    fn test_matrepr_dense_matrix_outer() {
        let m = MatReprDenseMatrix::outer_product(&[1.0, 2.0], &[3.0, 4.0]);
        assert!((m.get(0, 0) - 3.0).abs() < 1e-10);
        assert!((m.get(1, 1) - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_dense_matrix_submatrix() {
        let m = MatReprDenseMatrix::from_vec(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let sub = m.submatrix(0, 0, 2, 2);
        assert_eq!(sub.rows, 2);
        assert!((sub.get(1, 1) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_priority_queue() {
        let mut pq = MatReprPriorityQueue::new();
        pq.push(3.0, 1); pq.push(1.0, 2); pq.push(2.0, 3);
        assert_eq!(pq.pop().unwrap().1, 2);
        assert_eq!(pq.pop().unwrap().1, 3);
        assert_eq!(pq.pop().unwrap().1, 1);
    }

    #[test]
    fn test_matrepr_accumulator() {
        let mut acc = MatReprAccumulator::new();
        for i in 1..=10 { acc.add(i as f64); }
        assert!((acc.mean() - 5.5).abs() < 1e-10);
        assert_eq!(acc.count(), 10);
        assert!((acc.min() - 1.0).abs() < 1e-10);
        assert!((acc.max() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_accumulator_merge() {
        let mut a = MatReprAccumulator::new();
        a.add(1.0); a.add(2.0);
        let mut b = MatReprAccumulator::new();
        b.add(3.0); b.add(4.0);
        a.merge(&b);
        assert_eq!(a.count(), 4);
        assert!((a.mean() - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_sparse_matrix() {
        let mut m = MatReprSparseMatrix::new(3, 3);
        m.insert(0, 1, 2.0); m.insert(1, 2, 3.0);
        assert_eq!(m.nnz(), 2);
        assert!((m.get(0, 1) - 2.0).abs() < 1e-10);
        assert!((m.get(0, 0)).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_sparse_mul_vec() {
        let mut m = MatReprSparseMatrix::new(2, 2);
        m.insert(0, 0, 1.0); m.insert(1, 1, 2.0);
        let result = m.mul_vec(&[3.0, 4.0]);
        assert!((result[0] - 3.0).abs() < 1e-10);
        assert!((result[1] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_sparse_transpose() {
        let mut m = MatReprSparseMatrix::new(2, 3);
        m.insert(0, 2, 5.0);
        let t = m.transpose();
        assert_eq!(t.rows, 3); assert_eq!(t.cols, 2);
        assert!((t.get(2, 0) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_polynomial_eval() {
        let p = MatReprPolynomial::new(vec![1.0, 2.0, 3.0]);
        assert!((p.evaluate(2.0) - 17.0).abs() < 1e-10);
        assert!((p.evaluate_horner(2.0) - 17.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_polynomial_add() {
        let a = MatReprPolynomial::new(vec![1.0, 2.0]);
        let b = MatReprPolynomial::new(vec![3.0, 4.0]);
        let c = a.add(&b);
        assert!((c.evaluate(1.0) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_polynomial_mul() {
        let a = MatReprPolynomial::new(vec![1.0, 1.0]);
        let b = MatReprPolynomial::new(vec![1.0, 1.0]);
        let c = a.mul(&b);
        assert_eq!(c.degree(), 2);
        assert!((c.evaluate(2.0) - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_polynomial_deriv() {
        let p = MatReprPolynomial::new(vec![1.0, 2.0, 3.0]);
        let dp = p.derivative();
        assert!((dp.evaluate(0.0) - 2.0).abs() < 1e-10);
        assert!((dp.evaluate(1.0) - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_polynomial_integral() {
        let p = MatReprPolynomial::new(vec![2.0, 3.0]);
        let ip = p.integral(0.0);
        assert!((ip.evaluate(1.0) - 3.5).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_polynomial_roots() {
        let p = MatReprPolynomial::new(vec![-6.0, 1.0, 1.0]);
        let roots = p.roots_quadratic();
        assert_eq!(roots.len(), 2);
    }

    #[test]
    fn test_matrepr_polynomial_newton() {
        let p = MatReprPolynomial::new(vec![-2.0, 0.0, 1.0]);
        let root = p.newton_root(1.0, 100, 1e-10);
        assert!(root.is_some());
        assert!((root.unwrap() - std::f64::consts::SQRT_2).abs() < 1e-6);
    }

    #[test]
    fn test_matrepr_polynomial_compose() {
        let p = MatReprPolynomial::new(vec![0.0, 0.0, 1.0]);
        let q = MatReprPolynomial::new(vec![1.0, 1.0]);
        let r = p.compose(&q);
        assert!((r.evaluate(2.0) - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_rng() {
        let mut rng = MatReprRng::new(42);
        let v1 = rng.next_u64();
        let v2 = rng.next_u64();
        assert_ne!(v1, v2);
        let f = rng.next_f64();
        assert!(f >= 0.0 && f < 1.0);
    }

    #[test]
    fn test_matrepr_rng_gaussian() {
        let mut rng = MatReprRng::new(123);
        let mut sum = 0.0;
        for _ in 0..1000 { sum += rng.next_gaussian(); }
        let mean = sum / 1000.0;
        assert!(mean.abs() < 0.2);
    }

    #[test]
    fn test_matrepr_timer() {
        let mut timer = MatReprTimer::new("test");
        timer.record(100); timer.record(200); timer.record(300);
        assert_eq!(timer.count(), 3);
        assert_eq!(timer.total_ns(), 600);
        assert!((timer.average_ns() - 200.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_bitvector() {
        let mut bv = MatReprBitVector::new(100);
        bv.set(5); bv.set(42); bv.set(99);
        assert!(bv.get(5));
        assert!(bv.get(42));
        assert!(!bv.get(50));
        assert_eq!(bv.count_ones(), 3);
    }

    #[test]
    fn test_matrepr_bitvector_ops() {
        let mut a = MatReprBitVector::new(64);
        a.set(0); a.set(10); a.set(20);
        let mut b = MatReprBitVector::new(64);
        b.set(10); b.set(20); b.set(30);
        let c = a.and(&b);
        assert_eq!(c.count_ones(), 2);
        let d = a.or(&b);
        assert_eq!(d.count_ones(), 4);
        assert_eq!(a.hamming_distance(&b), 2);
    }

    #[test]
    fn test_matrepr_bitvector_jaccard() {
        let mut a = MatReprBitVector::new(10);
        a.set(0); a.set(1); a.set(2);
        let mut b = MatReprBitVector::new(10);
        b.set(1); b.set(2); b.set(3);
        let j = a.jaccard(&b);
        assert!((j - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_priority_queue_empty() {
        let mut pq = MatReprPriorityQueue::new();
        assert!(pq.is_empty());
        assert!(pq.pop().is_none());
    }

    #[test]
    fn test_matrepr_sparse_add() {
        let mut a = MatReprSparseMatrix::new(2, 2);
        a.insert(0, 0, 1.0);
        let mut b = MatReprSparseMatrix::new(2, 2);
        b.insert(0, 0, 2.0); b.insert(1, 1, 3.0);
        let c = a.add(&b);
        assert!((c.get(0, 0) - 3.0).abs() < 1e-10);
        assert!((c.get(1, 1) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_rng_shuffle() {
        let mut rng = MatReprRng::new(99);
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        rng.shuffle(&mut data);
        assert_eq!(data.len(), 5);
        let sum: f64 = data.iter().sum();
        assert!((sum - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_polynomial_display() {
        let p = MatReprPolynomial::new(vec![1.0, 2.0, 3.0]);
        let s = format!("{}", p);
        assert!(!s.is_empty());
    }

    #[test]
    fn test_matrepr_polynomial_monomial() {
        let m = MatReprPolynomial::monomial(3, 5.0);
        assert_eq!(m.degree(), 3);
        assert!((m.evaluate(2.0) - 40.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_timer_percentiles() {
        let mut timer = MatReprTimer::new("perf");
        for i in 1..=100 { timer.record(i); }
        assert_eq!(timer.p50_ns(), 50);
        assert!(timer.p95_ns() >= 90);
    }

    #[test]
    fn test_matrepr_accumulator_cv() {
        let mut acc = MatReprAccumulator::new();
        acc.add(10.0); acc.add(10.0); acc.add(10.0);
        assert!(acc.coefficient_of_variation().abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_sparse_diagonal() {
        let mut m = MatReprSparseMatrix::new(3, 3);
        m.insert(0, 0, 1.0); m.insert(1, 1, 2.0); m.insert(2, 2, 3.0);
        assert!((m.trace() - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_lru_cache() {
        let mut cache = MatReprLruCache::new(2);
        cache.put(1, vec![1.0]);
        cache.put(2, vec![2.0]);
        assert!(cache.get(1).is_some());
        cache.put(3, vec![3.0]);
        assert!(cache.get(2).is_none());
        assert!(cache.get(3).is_some());
    }

    #[test]
    fn test_matrepr_lru_hit_rate() {
        let mut cache = MatReprLruCache::new(10);
        cache.put(1, vec![1.0]);
        cache.get(1);
        cache.get(2);
        assert!((cache.hit_rate() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_graph_coloring() {
        let mut gc = MatReprGraphColoring::new(4);
        gc.add_edge(0, 1); gc.add_edge(1, 2); gc.add_edge(2, 3); gc.add_edge(3, 0);
        let colors = gc.greedy_color();
        assert!(gc.is_valid_coloring());
        assert!(colors <= 3);
    }

    #[test]
    fn test_matrepr_graph_coloring_complete() {
        let mut gc = MatReprGraphColoring::new(3);
        gc.add_edge(0, 1); gc.add_edge(1, 2); gc.add_edge(0, 2);
        let colors = gc.greedy_color();
        assert_eq!(colors, 3);
        assert!(gc.is_valid_coloring());
    }

    #[test]
    fn test_matrepr_topk() {
        let mut tk = MatReprTopK::new(3);
        tk.insert(5.0, "e"); tk.insert(3.0, "c"); tk.insert(1.0, "a");
        tk.insert(4.0, "d"); tk.insert(2.0, "b");
        assert_eq!(tk.len(), 3);
        assert!((tk.max_score().unwrap() - 5.0).abs() < 1e-10);
        assert!((tk.min_score().unwrap() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_sliding_window() {
        let mut sw = MatReprSlidingWindow::new(3);
        sw.push(1.0); sw.push(2.0); sw.push(3.0);
        assert!((sw.mean() - 2.0).abs() < 1e-10);
        sw.push(4.0);
        assert!((sw.mean() - 3.0).abs() < 1e-10);
        assert_eq!(sw.len(), 3);
    }

    #[test]
    fn test_matrepr_sliding_window_trend() {
        let mut sw = MatReprSlidingWindow::new(10);
        for i in 0..5 { sw.push(i as f64); }
        assert!(sw.trend() > 0.0);
    }

    #[test]
    fn test_matrepr_confusion_matrix() {
        let actual = vec![true, true, false, false, true];
        let predicted = vec![true, false, false, true, true];
        let cm = MatReprConfusionMatrix::from_predictions(&actual, &predicted);
        assert_eq!(cm.true_positive, 2);
        assert_eq!(cm.false_positive, 1);
        assert_eq!(cm.true_negative, 1);
        assert_eq!(cm.false_negative, 1);
        assert_eq!(cm.total(), 5);
    }

    #[test]
    fn test_matrepr_confusion_f1() {
        let cm = MatReprConfusionMatrix { true_positive: 80, false_positive: 20, true_negative: 70, false_negative: 30 };
        assert!((cm.precision() - 0.8).abs() < 1e-10);
        let f1 = cm.f1_score();
        assert!(f1 > 0.0 && f1 < 1.0);
    }

    #[test]
    fn test_matrepr_cosine_similarity() {
        let s = matrepr_cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]);
        assert!(s.abs() < 1e-10);
        let s2 = matrepr_cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]);
        assert!((s2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_euclidean_distance() {
        let d = matrepr_euclidean_distance(&[0.0, 0.0], &[3.0, 4.0]);
        assert!((d - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_sigmoid() {
        let s = matrepr_sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-10);
        let s2 = matrepr_sigmoid(100.0);
        assert!((s2 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_matrepr_softmax() {
        let sm = matrepr_softmax(&[1.0, 2.0, 3.0]);
        let sum: f64 = sm.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(sm[2] > sm[1] && sm[1] > sm[0]);
    }

    #[test]
    fn test_matrepr_kl_divergence() {
        let p = vec![0.5, 0.5];
        let q = vec![0.5, 0.5];
        let kl = matrepr_kl_divergence(&p, &q);
        assert!(kl.abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_normalize() {
        let v = matrepr_normalize(&[3.0, 4.0]);
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_lerp() {
        assert!((matrepr_lerp(0.0, 10.0, 0.5) - 5.0).abs() < 1e-10);
        assert!((matrepr_lerp(0.0, 10.0, 0.0) - 0.0).abs() < 1e-10);
        assert!((matrepr_lerp(0.0, 10.0, 1.0) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_clamp() {
        assert!((matrepr_clamp(5.0, 0.0, 10.0) - 5.0).abs() < 1e-10);
        assert!((matrepr_clamp(-5.0, 0.0, 10.0) - 0.0).abs() < 1e-10);
        assert!((matrepr_clamp(15.0, 0.0, 10.0) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_cross_product() {
        let c = matrepr_cross_product(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]);
        assert!((c[0]).abs() < 1e-10);
        assert!((c[1]).abs() < 1e-10);
        assert!((c[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_dot_product() {
        let d = matrepr_dot_product(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        assert!((d - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_js_divergence() {
        let p = vec![0.5, 0.5];
        let q = vec![0.5, 0.5];
        let js = matrepr_js_divergence(&p, &q);
        assert!(js.abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_hellinger() {
        let p = vec![0.5, 0.5];
        let q = vec![0.5, 0.5];
        let h = matrepr_hellinger_distance(&p, &q);
        assert!(h.abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_logsumexp() {
        let lse = matrepr_logsumexp(&[1.0, 2.0, 3.0]);
        assert!(lse > 3.0);
    }

    #[test]
    fn test_matrepr_feature_scaler() {
        let mut scaler = MatReprFeatureScaler::new();
        let data = vec![vec![1.0, 10.0], vec![2.0, 20.0], vec![3.0, 30.0]];
        scaler.fit(&data);
        let normalized = scaler.normalize(&[2.0, 20.0]);
        assert!((normalized[0] - 0.5).abs() < 1e-10);
        assert!((normalized[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_feature_scaler_inverse() {
        let mut scaler = MatReprFeatureScaler::new();
        let data = vec![vec![0.0, 0.0], vec![10.0, 100.0]];
        scaler.fit(&data);
        let normed = scaler.normalize(&[5.0, 50.0]);
        let inv = scaler.inverse_normalize(&normed);
        assert!((inv[0] - 5.0).abs() < 1e-10);
        assert!((inv[1] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_linear_regression() {
        let mut lr = MatReprLinearRegression::new();
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        lr.fit(&x, &y);
        assert!((lr.slope - 2.0).abs() < 1e-10);
        assert!(lr.intercept.abs() < 1e-10);
        assert!((lr.r_squared - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_linear_regression_predict() {
        let mut lr = MatReprLinearRegression::new();
        lr.fit(&[0.0, 1.0, 2.0], &[1.0, 3.0, 5.0]);
        assert!((lr.predict(3.0) - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_weighted_graph() {
        let mut g = MatReprWeightedGraph::new(4);
        g.add_edge(0, 1, 1.0); g.add_edge(1, 2, 2.0); g.add_edge(2, 3, 3.0);
        assert_eq!(g.num_edges, 3);
        let dists = g.dijkstra(0);
        assert!((dists[3] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_weighted_graph_mst() {
        let mut g = MatReprWeightedGraph::new(4);
        g.add_edge(0, 1, 1.0); g.add_edge(1, 2, 2.0); g.add_edge(2, 3, 3.0);
        g.add_edge(0, 3, 10.0);
        let mst = g.min_spanning_tree_weight();
        assert!((mst - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_moving_average() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ma = matrepr_moving_average(&data, 3);
        assert_eq!(ma.len(), 3);
        assert!((ma[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_cumsum() {
        let cs = matrepr_cumsum(&[1.0, 2.0, 3.0, 4.0]);
        assert!((cs[3] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_diff() {
        let d = matrepr_diff(&[1.0, 3.0, 6.0, 10.0]);
        assert_eq!(d.len(), 3);
        assert!((d[0] - 2.0).abs() < 1e-10);
        assert!((d[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_autocorrelation() {
        let ac = matrepr_autocorrelation(&[1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0], 0);
        assert!((ac - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_dft_magnitude() {
        let mags = matrepr_dft_magnitude(&[1.0, 0.0, -1.0, 0.0]);
        assert!(!mags.is_empty());
    }

    #[test]
    fn test_matrepr_integrate_trapezoid() {
        let area = matrepr_integrate_trapezoid(&[0.0, 1.0, 2.0], &[0.0, 1.0, 0.0]);
        assert!((area - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_convolve() {
        let c = matrepr_convolve(&[1.0, 2.0, 3.0], &[1.0, 1.0]);
        assert_eq!(c.len(), 4);
        assert!((c[0] - 1.0).abs() < 1e-10);
        assert!((c[3] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_weighted_graph_clustering() {
        let mut g = MatReprWeightedGraph::new(4);
        g.add_edge(0, 1, 1.0); g.add_edge(1, 2, 1.0); g.add_edge(0, 2, 1.0);
        let cc = g.clustering_coefficient(0);
        assert!((cc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_histogram() {
        let h = MatReprHistogramExt::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0], 5);
        assert_eq!(h.total, 5);
        assert_eq!(h.num_bins(), 5);
    }

    #[test]
    fn test_matrepr_histogram_cumulative() {
        let h = MatReprHistogramExt::from_data(&[1.0, 1.0, 2.0, 3.0, 3.0, 3.0], 3);
        let cum = h.cumulative();
        assert_eq!(*cum.last().unwrap(), 6);
    }

    #[test]
    fn test_matrepr_histogram_entropy() {
        let h = MatReprHistogramExt::from_data(&[1.0, 2.0, 3.0, 4.0], 4);
        let ent = h.entropy();
        assert!(ent > 0.0);
    }

    #[test]
    fn test_matrepr_aabb() {
        let bb = MatReprAABB::new(0.0, 0.0, 10.0, 10.0);
        assert!(bb.contains(5.0, 5.0));
        assert!(!bb.contains(11.0, 5.0));
        assert!((bb.area() - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_aabb_intersects() {
        let a = MatReprAABB::new(0.0, 0.0, 10.0, 10.0);
        let b = MatReprAABB::new(5.0, 5.0, 15.0, 15.0);
        let c = MatReprAABB::new(20.0, 20.0, 30.0, 30.0);
        assert!(a.intersects(&b));
        assert!(!a.intersects(&c));
    }

    #[test]
    fn test_matrepr_quadtree() {
        let bb = MatReprAABB::new(0.0, 0.0, 100.0, 100.0);
        let mut qt = MatReprQuadTree::new(bb, 4, 8);
        for i in 0..20 {
            qt.insert(MatReprPoint2D { x: i as f64 * 5.0, y: i as f64 * 5.0, data: i as f64 });
        }
        assert_eq!(qt.count(), 20);
    }

    #[test]
    fn test_matrepr_quadtree_query() {
        let bb = MatReprAABB::new(0.0, 0.0, 100.0, 100.0);
        let mut qt = MatReprQuadTree::new(bb, 2, 8);
        qt.insert(MatReprPoint2D { x: 10.0, y: 10.0, data: 1.0 });
        qt.insert(MatReprPoint2D { x: 90.0, y: 90.0, data: 2.0 });
        let range = MatReprAABB::new(0.0, 0.0, 50.0, 50.0);
        let found = qt.query_range(&range);
        assert_eq!(found.len(), 1);
    }

    #[test]
    fn test_matrepr_mat_mul() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
        let c = matrepr_mat_mul(&a, &b);
        assert!((c[0][0] - 19.0).abs() < 1e-10);
        assert!((c[1][1] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_transpose() {
        let a = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let t = matrepr_transpose(&a);
        assert_eq!(t.len(), 3);
        assert_eq!(t[0].len(), 2);
        assert!((t[2][1] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_frobenius_norm() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let n = matrepr_frobenius_norm(&a);
        assert!((n - 2.0f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_trace() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        assert!((matrepr_trace(&a) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_identity() {
        let id = matrepr_identity(3);
        assert!((id[0][0] - 1.0).abs() < 1e-10);
        assert!((id[0][1]).abs() < 1e-10);
        assert!((id[2][2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_power_iteration() {
        let a = vec![vec![2.0, 1.0], vec![1.0, 2.0]];
        let (eval, _evec) = matrepr_power_iteration(&a, 100);
        assert!((eval - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_matrepr_running_stats() {
        let mut s = MatReprRunningStats::new();
        for &v in &[1.0, 2.0, 3.0, 4.0, 5.0] { s.push(v); }
        assert!((s.mean - 3.0).abs() < 1e-10);
        assert!((s.min_val - 1.0).abs() < 1e-10);
        assert!((s.max_val - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_running_stats_merge() {
        let mut a = MatReprRunningStats::new();
        let mut b = MatReprRunningStats::new();
        for &v in &[1.0, 2.0, 3.0] { a.push(v); }
        for &v in &[4.0, 5.0, 6.0] { b.push(v); }
        a.merge(&b);
        assert_eq!(a.count, 6);
        assert!((a.mean - 3.5).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_running_stats_cv() {
        let mut s = MatReprRunningStats::new();
        for &v in &[10.0, 10.0, 10.0] { s.push(v); }
        assert!(s.coefficient_of_variation() < 1e-10);
    }

    #[test]
    fn test_matrepr_percentile_at() {
        let p50 = matrepr_percentile_at(&[1.0, 2.0, 3.0, 4.0, 5.0], 50.0);
        assert!((p50 - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_iqr() {
        let iqr = matrepr_iqr(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        assert!(iqr > 0.0);
    }

    #[test]
    fn test_matrepr_outliers() {
        let outliers = matrepr_outliers(&[1.0, 2.0, 3.0, 4.0, 5.0, 100.0]);
        assert!(!outliers.is_empty());
    }

    #[test]
    fn test_matrepr_zscore() {
        let z = matrepr_zscore(&[10.0, 20.0, 30.0]);
        assert!((z[1]).abs() < 1e-10); // middle value should be ~0
    }

    #[test]
    fn test_matrepr_rank() {
        let r = matrepr_rank(&[30.0, 10.0, 20.0]);
        assert!((r[0] - 3.0).abs() < 1e-10);
        assert!((r[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_spearman() {
        let rho = matrepr_spearman(&[1.0, 2.0, 3.0, 4.0, 5.0], &[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!((rho - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_geomean() {
        let gm = matrepr_geomean(&[1.0, 2.0, 4.0, 8.0]);
        assert!((gm - (1.0 * 2.0 * 4.0 * 8.0_f64).powf(0.25)).abs() < 1e-6);
    }

    #[test]
    fn test_matrepr_harmmean() {
        let hm = matrepr_harmmean(&[1.0, 2.0, 4.0]);
        let expected = 3.0 / (1.0 + 0.5 + 0.25);
        assert!((hm - expected).abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_sample_skewness_symmetric() {
        let s = matrepr_sample_skewness(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(s.abs() < 1e-10);
    }

    #[test]
    fn test_matrepr_excess_kurtosis() {
        let k = matrepr_excess_kurtosis(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(k.is_finite());
    }

    #[test]
    fn test_matrepr_covariance_matrix() {
        let data = vec![vec![1.0, 2.0], vec![2.0, 4.0], vec![3.0, 6.0]];
        let cov = matrepr_covariance_matrix(&data);
        assert_eq!(cov.len(), 2);
        assert!(cov[0][0] > 0.0);
    }

    #[test]
    fn test_matrepr_correlation_matrix() {
        let data = vec![vec![1.0, 2.0], vec![2.0, 4.0], vec![3.0, 6.0]];
        let corr = matrepr_correlation_matrix(&data);
        assert!((corr[0][1] - 1.0).abs() < 1e-10); // perfect correlation
    }

}
