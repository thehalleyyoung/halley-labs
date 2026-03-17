//! Geometric structures for phase space analysis.

use serde::{Deserialize, Serialize};
use std::fmt;

/// A differential form on the phase space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DifferentialForm {
    pub degree: usize,
    pub dimension: usize,
    pub components: Vec<FormComponent>,
}

impl DifferentialForm {
    pub fn new(degree: usize, dimension: usize) -> Self {
        Self { degree, dimension, components: Vec::new() }
    }

    pub fn zero_form(dimension: usize) -> Self {
        Self::new(0, dimension)
    }

    pub fn one_form(dimension: usize, coefficients: Vec<f64>) -> Self {
        let components = coefficients.into_iter().enumerate().map(|(i, c)| {
            FormComponent { indices: vec![i], coefficient: c }
        }).collect();
        Self { degree: 1, dimension, components }
    }

    pub fn two_form(dimension: usize, coefficients: Vec<(usize, usize, f64)>) -> Self {
        let components = coefficients.into_iter().map(|(i, j, c)| {
            FormComponent { indices: vec![i, j], coefficient: c }
        }).collect();
        Self { degree: 2, dimension, components }
    }

    pub fn wedge(&self, other: &DifferentialForm) -> DifferentialForm {
        let new_degree = self.degree + other.degree;
        let mut components = Vec::new();
        for a in &self.components {
            for b in &other.components {
                let mut indices = a.indices.clone();
                indices.extend(&b.indices);
                let has_repeat = {
                    let mut seen = std::collections::HashSet::new();
                    indices.iter().any(|i| !seen.insert(i))
                };
                if !has_repeat {
                    let sign = permutation_sign(&indices);
                    let mut sorted_indices = indices.clone();
                    sorted_indices.sort();
                    components.push(FormComponent {
                        indices: sorted_indices,
                        coefficient: sign as f64 * a.coefficient * b.coefficient,
                    });
                }
            }
        }
        DifferentialForm { degree: new_degree, dimension: self.dimension, components }
    }

    pub fn exterior_derivative(&self) -> DifferentialForm {
        let new_degree = self.degree + 1;
        let mut components = Vec::new();
        for comp in &self.components {
            for d in 0..self.dimension {
                if !comp.indices.contains(&d) {
                    let mut new_indices = vec![d];
                    new_indices.extend(&comp.indices);
                    let sign = permutation_sign(&new_indices);
                    let mut sorted = new_indices;
                    sorted.sort();
                    components.push(FormComponent {
                        indices: sorted,
                        coefficient: sign as f64 * comp.coefficient,
                    });
                }
            }
        }
        consolidate_components(&mut components);
        DifferentialForm { degree: new_degree, dimension: self.dimension, components }
    }

    pub fn evaluate(&self, vectors: &[Vec<f64>]) -> f64 {
        if vectors.len() != self.degree { return 0.0; }
        let mut result = 0.0;
        for comp in &self.components {
            let mut term = comp.coefficient;
            for (k, &idx) in comp.indices.iter().enumerate() {
                if k < vectors.len() && idx < vectors[k].len() {
                    term *= vectors[k][idx];
                } else {
                    term = 0.0;
                    break;
                }
            }
            result += term;
        }
        result
    }

    pub fn is_closed(&self) -> bool {
        let d_self = self.exterior_derivative();
        d_self.components.iter().all(|c| c.coefficient.abs() < 1e-12)
    }
}

/// A component of a differential form.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormComponent {
    pub indices: Vec<usize>,
    pub coefficient: f64,
}

fn permutation_sign(perm: &[usize]) -> i32 {
    let n = perm.len();
    let mut sign = 1i32;
    let mut sorted = perm.to_vec();
    for i in 0..n {
        for j in i + 1..n {
            if sorted[i] > sorted[j] {
                sorted.swap(i, j);
                sign = -sign;
            }
        }
    }
    sign
}

fn consolidate_components(components: &mut Vec<FormComponent>) {
    let mut map: std::collections::HashMap<Vec<usize>, f64> = std::collections::HashMap::new();
    for comp in components.iter() {
        *map.entry(comp.indices.clone()).or_default() += comp.coefficient;
    }
    *components = map.into_iter()
        .filter(|(_, c)| c.abs() > 1e-15)
        .map(|(indices, coefficient)| FormComponent { indices, coefficient })
        .collect();
}

/// A vector field on the phase space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorField {
    pub dimension: usize,
    pub components: Vec<VectorFieldComponent>,
}

impl VectorField {
    pub fn new(dimension: usize) -> Self {
        Self { dimension, components: Vec::new() }
    }

    pub fn constant(values: Vec<f64>) -> Self {
        let dimension = values.len();
        let components = values.into_iter().enumerate().map(|(i, v)| {
            VectorFieldComponent {
                index: i,
                coefficient_type: CoefficientType::Constant(v),
            }
        }).collect();
        Self { dimension, components }
    }

    pub fn evaluate_at(&self, point: &[f64]) -> Vec<f64> {
        let mut result = vec![0.0; self.dimension];
        for comp in &self.components {
            if comp.index < self.dimension {
                result[comp.index] = comp.coefficient_type.evaluate(point);
            }
        }
        result
    }

    pub fn lie_bracket(&self, other: &VectorField) -> VectorField {
        let dim = self.dimension;
        let h = 1e-8;
        let mut components = Vec::new();
        let origin = vec![0.0; dim];
        let v_at_origin = self.evaluate_at(&origin);
        let w_at_origin = other.evaluate_at(&origin);
        for i in 0..dim {
            let mut bracket_val = 0.0;
            for j in 0..dim {
                let mut point_plus = origin.clone();
                point_plus[j] += h;
                let v_shifted = self.evaluate_at(&point_plus);
                let w_shifted = other.evaluate_at(&point_plus);
                let dv_dj = (v_shifted[i] - v_at_origin[i]) / h;
                let dw_dj = (w_shifted[i] - w_at_origin[i]) / h;
                bracket_val += w_at_origin[j] * dv_dj - v_at_origin[j] * dw_dj;
            }
            components.push(VectorFieldComponent {
                index: i,
                coefficient_type: CoefficientType::Constant(bracket_val),
            });
        }
        VectorField { dimension: dim, components }
    }

    pub fn divergence_at(&self, point: &[f64]) -> f64 {
        let h = 1e-8;
        let mut div = 0.0;
        for i in 0..self.dimension {
            let mut point_plus = point.to_vec();
            let mut point_minus = point.to_vec();
            point_plus[i] += h;
            point_minus[i] -= h;
            let v_plus = self.evaluate_at(&point_plus);
            let v_minus = self.evaluate_at(&point_minus);
            div += (v_plus[i] - v_minus[i]) / (2.0 * h);
        }
        div
    }
}

/// A component of a vector field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorFieldComponent {
    pub index: usize,
    pub coefficient_type: CoefficientType,
}

/// Type of coefficient for vector field components.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoefficientType {
    Constant(f64),
    Linear(Vec<f64>),
    Polynomial(Vec<(Vec<usize>, f64)>),
}

impl CoefficientType {
    pub fn evaluate(&self, point: &[f64]) -> f64 {
        match self {
            CoefficientType::Constant(c) => *c,
            CoefficientType::Linear(coeffs) => {
                coeffs.iter().enumerate().map(|(i, c)| c * point.get(i).unwrap_or(&0.0)).sum()
            }
            CoefficientType::Polynomial(terms) => {
                terms.iter().map(|(powers, coeff)| {
                    let prod: f64 = powers.iter().enumerate().map(|(i, &p)| {
                        point.get(i).unwrap_or(&0.0).powi(p as i32)
                    }).product();
                    coeff * prod
                }).sum()
            }
        }
    }
}

/// A Poisson bracket structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoissonBracket {
    pub dimension: usize,
    pub matrix: Vec<Vec<f64>>,
}

impl PoissonBracket {
    pub fn canonical(n: usize) -> Self {
        let dim = 2 * n;
        let mut matrix = vec![vec![0.0; dim]; dim];
        for i in 0..n {
            matrix[i][i + n] = 1.0;
            matrix[i + n][i] = -1.0;
        }
        Self { dimension: dim, matrix }
    }

    pub fn evaluate(&self, df: &[f64], dg: &[f64]) -> f64 {
        let mut result = 0.0;
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                result += self.matrix[i][j] * df[i] * dg[j];
            }
        }
        result
    }

    pub fn is_antisymmetric(&self) -> bool {
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                if (self.matrix[i][j] + self.matrix[j][i]).abs() > 1e-12 {
                    return false;
                }
            }
        }
        true
    }

    pub fn satisfies_jacobi(&self, test_points: &[Vec<f64>]) -> bool {
        for _point in test_points {
            let n = self.dimension;
            for i in 0..n {
                for j in 0..n {
                    for k in 0..n {
                        let mut jacobi = 0.0;
                        for l in 0..n {
                            jacobi += self.matrix[i][l] * self.matrix[j][k];
                            jacobi += self.matrix[j][l] * self.matrix[k][i];
                            jacobi += self.matrix[k][l] * self.matrix[i][j];
                        }
                        if jacobi.abs() > 1e-8 {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_one_form() {
        let form = DifferentialForm::one_form(3, vec![1.0, 0.0, 0.0]);
        assert_eq!(form.degree, 1);
        assert_eq!(form.components.len(), 3);
    }

    #[test]
    fn test_wedge_product() {
        let dx = DifferentialForm::one_form(3, vec![1.0, 0.0, 0.0]);
        let dy = DifferentialForm::one_form(3, vec![0.0, 1.0, 0.0]);
        let dxdy = dx.wedge(&dy);
        assert_eq!(dxdy.degree, 2);
    }

    #[test]
    fn test_vector_field_evaluate() {
        let vf = VectorField::constant(vec![1.0, 2.0, 3.0]);
        let result = vf.evaluate_at(&[0.0, 0.0, 0.0]);
        assert!((result[0] - 1.0).abs() < 1e-12);
        assert!((result[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_poisson_bracket() {
        let pb = PoissonBracket::canonical(2);
        assert!(pb.is_antisymmetric());
        let df = vec![1.0, 0.0, 0.0, 0.0];
        let dg = vec![0.0, 0.0, 1.0, 0.0];
        let result = pb.evaluate(&df, &dg);
        assert!((result - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_permutation_sign() {
        assert_eq!(permutation_sign(&[0, 1, 2]), 1);
        assert_eq!(permutation_sign(&[1, 0, 2]), -1);
        assert_eq!(permutation_sign(&[2, 0, 1]), 1);
    }
}
