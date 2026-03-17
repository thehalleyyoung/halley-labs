//! ndarray interop: conversions between sim-types and ndarray Array types.

#[cfg(feature = "ndarray-interop")]
mod ndarray_impl {
    use crate::vector::Vec3;

    /// Convert a slice of Vec3 positions to an ndarray Array2<f64> with shape (N, 3).
    pub fn positions_to_array(positions: &[Vec3]) -> ndarray::Array2<f64> {
        let n = positions.len();
        let mut arr = ndarray::Array2::<f64>::zeros((n, 3));
        for (i, p) in positions.iter().enumerate() {
            arr[[i, 0]] = p.x;
            arr[[i, 1]] = p.y;
            arr[[i, 2]] = p.z;
        }
        arr
    }

    /// Convert an ndarray Array2<f64> with shape (N, 3) to a Vec of Vec3.
    pub fn array_to_positions(arr: &ndarray::Array2<f64>) -> Vec<Vec3> {
        let n = arr.nrows();
        let mut positions = Vec::with_capacity(n);
        for i in 0..n {
            positions.push(Vec3::new(arr[[i, 0]], arr[[i, 1]], arr[[i, 2]]));
        }
        positions
    }

    /// Convert a Vec of f64 values to a 1D ndarray.
    pub fn values_to_array1(values: &[f64]) -> ndarray::Array1<f64> {
        ndarray::Array1::from_vec(values.to_vec())
    }

    /// Convert a time series of conserved quantities to an ndarray Array2<f64>
    /// with shape (num_steps, num_quantities).
    pub fn time_series_to_array(series: &[Vec<f64>]) -> ndarray::Array2<f64> {
        if series.is_empty() {
            return ndarray::Array2::<f64>::zeros((0, 0));
        }
        let nrows = series.len();
        let ncols = series[0].len();
        let mut arr = ndarray::Array2::<f64>::zeros((nrows, ncols));
        for (i, row) in series.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                if j < ncols {
                    arr[[i, j]] = val;
                }
            }
        }
        arr
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn positions_roundtrip() {
            let positions = vec![
                Vec3::new(1.0, 2.0, 3.0),
                Vec3::new(4.0, 5.0, 6.0),
                Vec3::new(7.0, 8.0, 9.0),
            ];
            let arr = positions_to_array(&positions);
            assert_eq!(arr.shape(), &[3, 3]);
            let back = array_to_positions(&arr);
            assert_eq!(back.len(), 3);
            assert!((back[0].x - 1.0).abs() < 1e-15);
            assert!((back[2].z - 9.0).abs() < 1e-15);
        }

        #[test]
        fn time_series_conversion() {
            let series = vec![
                vec![1.0, 2.0, 3.0],
                vec![4.0, 5.0, 6.0],
            ];
            let arr = time_series_to_array(&series);
            assert_eq!(arr.shape(), &[2, 3]);
            assert!((arr[[1, 2]] - 6.0).abs() < 1e-15);
        }
    }
}

#[cfg(feature = "ndarray-interop")]
pub use ndarray_impl::*;
