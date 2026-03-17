//! Symmetry group and Lie algebra types.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// A symmetry group of a dynamical system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymmetryGroup {
    pub name: String,
    pub dimension: usize,
    pub generators: Vec<SymmetryGenerator>,
    pub structure_constants: Vec<Vec<Vec<f64>>>,
    pub group_type: SymmetryGroupType,
    pub is_compact: bool,
}

impl SymmetryGroup {
    pub fn new(name: impl Into<String>, group_type: SymmetryGroupType) -> Self {
        Self {
            name: name.into(),
            dimension: 0,
            generators: Vec::new(),
            structure_constants: Vec::new(),
            group_type,
            is_compact: false,
        }
    }

    pub fn translation(spatial_dim: usize) -> Self {
        let mut group = Self::new(
            format!("T({})", spatial_dim),
            SymmetryGroupType::Translation,
        );
        group.dimension = spatial_dim;
        for i in 0..spatial_dim {
            let names = ["x", "y", "z"];
            let name = names.get(i).unwrap_or(&"?");
            group.generators.push(SymmetryGenerator::translation(i, *name));
        }
        let d = spatial_dim;
        group.structure_constants = vec![vec![vec![0.0; d]; d]; d];
        group
    }

    pub fn rotation(spatial_dim: usize) -> Self {
        assert!(spatial_dim >= 2, "Rotation requires at least 2 dimensions");
        let gen_count = spatial_dim * (spatial_dim - 1) / 2;
        let mut group = Self::new(
            format!("SO({})", spatial_dim),
            SymmetryGroupType::Rotation,
        );
        group.dimension = gen_count;
        group.is_compact = true;

        if spatial_dim == 3 {
            for (i, name) in ["x", "y", "z"].iter().enumerate() {
                group.generators.push(SymmetryGenerator::rotation(i, name));
            }
            let mut sc = vec![vec![vec![0.0; 3]; 3]; 3];
            sc[0][1][2] = 1.0;
            sc[1][0][2] = -1.0;
            sc[1][2][0] = 1.0;
            sc[2][1][0] = -1.0;
            sc[0][2][1] = -1.0;
            sc[2][0][1] = 1.0;
            group.structure_constants = sc;
        } else {
            let mut idx = 0;
            for i in 0..spatial_dim {
                for j in (i + 1)..spatial_dim {
                    group
                        .generators
                        .push(SymmetryGenerator::rotation_plane(i, j, idx));
                    idx += 1;
                }
            }
            group.structure_constants = vec![vec![vec![0.0; gen_count]; gen_count]; gen_count];
        }
        group
    }

    pub fn time_translation() -> Self {
        let mut group = Self::new("T_t", SymmetryGroupType::TimeTranslation);
        group.dimension = 1;
        group.generators.push(SymmetryGenerator {
            id: "time_translation".to_string(),
            name: "Time Translation".to_string(),
            kind: GeneratorKind::TimeTranslation,
            coefficients: vec![1.0],
            phase_space_action: Vec::new(),
        });
        group.structure_constants = vec![vec![vec![0.0]]];
        group
    }

    pub fn galilean(spatial_dim: usize) -> Self {
        let mut group = Self::new(
            format!("Gal({})", spatial_dim),
            SymmetryGroupType::Galilean,
        );
        group.dimension = spatial_dim;
        for i in 0..spatial_dim {
            let names = ["x", "y", "z"];
            let name = names.get(i).unwrap_or(&"?");
            group.generators.push(SymmetryGenerator {
                id: format!("galilean_{}", name),
                name: format!("Galilean Boost ({})", name),
                kind: GeneratorKind::GalileanBoost(i),
                coefficients: vec![1.0],
                phase_space_action: Vec::new(),
            });
        }
        group.structure_constants =
            vec![vec![vec![0.0; spatial_dim]; spatial_dim]; spatial_dim];
        group
    }

    pub fn scaling() -> Self {
        let mut group = Self::new("Scale", SymmetryGroupType::Scaling);
        group.dimension = 1;
        group.generators.push(SymmetryGenerator {
            id: "scaling".to_string(),
            name: "Scaling".to_string(),
            kind: GeneratorKind::Scaling,
            coefficients: vec![1.0],
            phase_space_action: Vec::new(),
        });
        group.structure_constants = vec![vec![vec![0.0]]];
        group
    }

    pub fn lie_bracket(&self, i: usize, j: usize) -> Vec<f64> {
        if i >= self.dimension || j >= self.dimension {
            return vec![0.0; self.dimension];
        }
        let mut result = vec![0.0; self.dimension];
        for k in 0..self.dimension {
            result[k] = self.structure_constants[i][j][k];
        }
        result
    }

    pub fn is_abelian(&self) -> bool {
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                for k in 0..self.dimension {
                    if self.structure_constants[i][j][k].abs() > 1e-12 {
                        return false;
                    }
                }
            }
        }
        true
    }

    pub fn killing_form(&self) -> Vec<Vec<f64>> {
        let d = self.dimension;
        let mut kf = vec![vec![0.0; d]; d];
        for a in 0..d {
            for b in 0..d {
                let mut sum = 0.0;
                for c in 0..d {
                    for e in 0..d {
                        sum += self.structure_constants[a][c][e]
                            * self.structure_constants[b][e][c];
                    }
                }
                kf[a][b] = sum;
            }
        }
        kf
    }

    pub fn check_jacobi_identity(&self) -> bool {
        let d = self.dimension;
        for i in 0..d {
            for j in 0..d {
                for k in 0..d {
                    for l in 0..d {
                        let mut sum = 0.0;
                        for m in 0..d {
                            sum += self.structure_constants[i][j][m]
                                * self.structure_constants[m][k][l];
                            sum += self.structure_constants[j][k][m]
                                * self.structure_constants[m][i][l];
                            sum += self.structure_constants[k][i][m]
                                * self.structure_constants[m][j][l];
                        }
                        if sum.abs() > 1e-10 {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }
}

/// Types of symmetry groups.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SymmetryGroupType {
    Translation,
    Rotation,
    TimeTranslation,
    Galilean,
    Scaling,
    Poincare,
    Conformal,
    Gauge,
    Custom,
}

/// An infinitesimal generator of a symmetry group.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymmetryGenerator {
    pub id: String,
    pub name: String,
    pub kind: GeneratorKind,
    pub coefficients: Vec<f64>,
    pub phase_space_action: Vec<PhaseSpaceAction>,
}

impl SymmetryGenerator {
    pub fn translation(axis: usize, name: &str) -> Self {
        Self {
            id: format!("translation_{}", name),
            name: format!("Translation ({})", name),
            kind: GeneratorKind::Translation(axis),
            coefficients: vec![1.0],
            phase_space_action: vec![PhaseSpaceAction {
                coordinate_index: axis,
                action_type: ActionType::Shift,
                coefficient: 1.0,
            }],
        }
    }

    pub fn rotation(axis: usize, name: &str) -> Self {
        Self {
            id: format!("rotation_{}", name),
            name: format!("Rotation ({})", name),
            kind: GeneratorKind::Rotation(axis),
            coefficients: vec![1.0],
            phase_space_action: Vec::new(),
        }
    }

    pub fn rotation_plane(i: usize, j: usize, idx: usize) -> Self {
        Self {
            id: format!("rotation_{}_{}", i, j),
            name: format!("Rotation in {}-{} plane", i, j),
            kind: GeneratorKind::RotationPlane(i, j),
            coefficients: vec![1.0],
            phase_space_action: vec![
                PhaseSpaceAction {
                    coordinate_index: i,
                    action_type: ActionType::LinearCombination(vec![(j, 1.0)]),
                    coefficient: 1.0,
                },
                PhaseSpaceAction {
                    coordinate_index: j,
                    action_type: ActionType::LinearCombination(vec![(i, -1.0)]),
                    coefficient: 1.0,
                },
            ],
        }
    }

    pub fn apply_to_vector(&self, state: &[f64]) -> Vec<f64> {
        let n = state.len();
        let mut result = vec![0.0; n];
        for action in &self.phase_space_action {
            if action.coordinate_index < n {
                match &action.action_type {
                    ActionType::Shift => {
                        result[action.coordinate_index] += action.coefficient;
                    }
                    ActionType::Scale => {
                        result[action.coordinate_index] +=
                            action.coefficient * state[action.coordinate_index];
                    }
                    ActionType::LinearCombination(terms) => {
                        for &(idx, coeff) in terms {
                            if idx < n {
                                result[action.coordinate_index] +=
                                    action.coefficient * coeff * state[idx];
                            }
                        }
                    }
                }
            }
        }
        result
    }
}

/// Kinds of symmetry generators.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GeneratorKind {
    Translation(usize),
    Rotation(usize),
    RotationPlane(usize, usize),
    TimeTranslation,
    GalileanBoost(usize),
    Scaling,
    Reflection(usize),
    Custom(String),
}

/// Action of a generator on a phase space coordinate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseSpaceAction {
    pub coordinate_index: usize,
    pub action_type: ActionType,
    pub coefficient: f64,
}

/// Types of actions on coordinates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionType {
    Shift,
    Scale,
    LinearCombination(Vec<(usize, f64)>),
}

/// A Lie algebra structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LieAlgebra {
    pub name: String,
    pub dimension: usize,
    pub basis: Vec<LieBasisElement>,
    pub structure_constants: Vec<Vec<Vec<f64>>>,
    pub properties: LieAlgebraProperties,
}

impl LieAlgebra {
    pub fn new(name: impl Into<String>, dimension: usize) -> Self {
        Self {
            name: name.into(),
            dimension,
            basis: Vec::new(),
            structure_constants: vec![vec![vec![0.0; dimension]; dimension]; dimension],
            properties: LieAlgebraProperties::default(),
        }
    }

    pub fn set_structure_constant(&mut self, i: usize, j: usize, k: usize, value: f64) {
        if i < self.dimension && j < self.dimension && k < self.dimension {
            self.structure_constants[i][j][k] = value;
            self.structure_constants[j][i][k] = -value;
        }
    }

    pub fn bracket(&self, v: &[f64], w: &[f64]) -> Vec<f64> {
        let d = self.dimension;
        let mut result = vec![0.0; d];
        for i in 0..d {
            for j in 0..d {
                for k in 0..d {
                    result[k] += self.structure_constants[i][j][k] * v[i] * w[j];
                }
            }
        }
        result
    }

    pub fn adjoint(&self, x: &[f64]) -> Vec<Vec<f64>> {
        let d = self.dimension;
        let mut ad = vec![vec![0.0; d]; d];
        for j in 0..d {
            for k in 0..d {
                for i in 0..d {
                    ad[j][k] += self.structure_constants[i][j][k] * x[i];
                }
            }
        }
        ad
    }

    pub fn compute_properties(&mut self) {
        self.properties.is_abelian = self.check_abelian();
        self.properties.is_nilpotent = self.check_nilpotent(10);
        self.properties.is_solvable = self.check_solvable(10);
        self.properties.is_semisimple = self.check_semisimple();
    }

    fn check_abelian(&self) -> bool {
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                for k in 0..self.dimension {
                    if self.structure_constants[i][j][k].abs() > 1e-12 {
                        return false;
                    }
                }
            }
        }
        true
    }

    fn check_nilpotent(&self, max_steps: usize) -> bool {
        let d = self.dimension;
        if d == 0 {
            return true;
        }
        let mut current_basis: Vec<Vec<f64>> = (0..d)
            .map(|i| {
                let mut v = vec![0.0; d];
                v[i] = 1.0;
                v
            })
            .collect();
        for _ in 0..max_steps {
            let mut next_basis = Vec::new();
            for v in &current_basis {
                for i in 0..d {
                    let mut ei = vec![0.0; d];
                    ei[i] = 1.0;
                    let b = self.bracket(v, &ei);
                    let norm: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
                    if norm > 1e-12 {
                        next_basis.push(b);
                    }
                }
            }
            if next_basis.is_empty() {
                return true;
            }
            current_basis = next_basis;
        }
        false
    }

    fn check_solvable(&self, max_steps: usize) -> bool {
        let d = self.dimension;
        if d == 0 {
            return true;
        }
        let mut current_sc = self.structure_constants.clone();
        for _ in 0..max_steps {
            let mut all_zero = true;
            for i in 0..d {
                for j in 0..d {
                    for k in 0..d {
                        if current_sc[i][j][k].abs() > 1e-12 {
                            all_zero = false;
                        }
                    }
                }
            }
            if all_zero {
                return true;
            }
            let mut next_sc = vec![vec![vec![0.0; d]; d]; d];
            for a in 0..d {
                for b in 0..d {
                    for c in 0..d {
                        for i in 0..d {
                            for j in 0..d {
                                for k in 0..d {
                                    next_sc[a][b][c] +=
                                        current_sc[a][i][j] * current_sc[b][j][k]
                                            * if k == c { 1.0 } else { 0.0 };
                                }
                            }
                        }
                    }
                }
            }
            current_sc = next_sc;
        }
        false
    }

    fn check_semisimple(&self) -> bool {
        let d = self.dimension;
        if d == 0 {
            return false;
        }
        let kf = self.killing_form();
        let det = simple_determinant(&kf);
        det.abs() > 1e-10
    }

    pub fn killing_form(&self) -> Vec<Vec<f64>> {
        let d = self.dimension;
        let mut kf = vec![vec![0.0; d]; d];
        for a in 0..d {
            for b in 0..d {
                let ad_a = {
                    let mut m = vec![vec![0.0; d]; d];
                    for j in 0..d {
                        for k in 0..d {
                            m[j][k] = self.structure_constants[a][j][k];
                        }
                    }
                    m
                };
                let ad_b = {
                    let mut m = vec![vec![0.0; d]; d];
                    for j in 0..d {
                        for k in 0..d {
                            m[j][k] = self.structure_constants[b][j][k];
                        }
                    }
                    m
                };
                let mut trace = 0.0;
                for i in 0..d {
                    for j in 0..d {
                        trace += ad_a[i][j] * ad_b[j][i];
                    }
                }
                kf[a][b] = trace;
            }
        }
        kf
    }
}

fn simple_determinant(mat: &[Vec<f64>]) -> f64 {
    let n = mat.len();
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return mat[0][0];
    }
    if n == 2 {
        return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0];
    }
    let mut result = 0.0;
    for j in 0..n {
        let minor: Vec<Vec<f64>> = (1..n)
            .map(|i| {
                (0..n)
                    .filter(|&k| k != j)
                    .map(|k| mat[i][k])
                    .collect()
            })
            .collect();
        let sign = if j % 2 == 0 { 1.0 } else { -1.0 };
        result += sign * mat[0][j] * simple_determinant(&minor);
    }
    result
}

/// Properties of a Lie algebra.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LieAlgebraProperties {
    pub is_abelian: bool,
    pub is_nilpotent: bool,
    pub is_solvable: bool,
    pub is_semisimple: bool,
    pub rank: Option<usize>,
    pub cartan_type: Option<String>,
}

/// A basis element of a Lie algebra.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LieBasisElement {
    pub index: usize,
    pub name: String,
    pub components: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_translation_group() {
        let g = SymmetryGroup::translation(3);
        assert_eq!(g.dimension, 3);
        assert!(g.is_abelian());
    }

    #[test]
    fn test_rotation_group() {
        let g = SymmetryGroup::rotation(3);
        assert_eq!(g.dimension, 3);
        assert!(!g.is_abelian());
        assert!(g.check_jacobi_identity());
    }

    #[test]
    fn test_so3_structure() {
        let g = SymmetryGroup::rotation(3);
        let bracket = g.lie_bracket(0, 1);
        assert!((bracket[2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_lie_algebra_bracket() {
        let mut la = LieAlgebra::new("so(3)", 3);
        la.set_structure_constant(0, 1, 2, 1.0);
        la.set_structure_constant(1, 2, 0, 1.0);
        la.set_structure_constant(2, 0, 1, 1.0);

        let e1 = vec![1.0, 0.0, 0.0];
        let e2 = vec![0.0, 1.0, 0.0];
        let result = la.bracket(&e1, &e2);
        assert!((result[2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_lie_algebra_properties() {
        let mut la = LieAlgebra::new("abelian", 2);
        la.compute_properties();
        assert!(la.properties.is_abelian);
        assert!(la.properties.is_nilpotent);
        assert!(la.properties.is_solvable);
    }

    #[test]
    fn test_killing_form() {
        let mut la = LieAlgebra::new("so(3)", 3);
        la.set_structure_constant(0, 1, 2, 1.0);
        la.set_structure_constant(1, 2, 0, 1.0);
        la.set_structure_constant(2, 0, 1, 1.0);
        let kf = la.killing_form();
        assert!((kf[0][0] - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_generator_action() {
        let gen = SymmetryGenerator::translation(0, "x");
        let state = vec![1.0, 2.0, 3.0];
        let result = gen.apply_to_vector(&state);
        assert!((result[0] - 1.0).abs() < 1e-12);
        assert!((result[1]).abs() < 1e-12);
    }
}
