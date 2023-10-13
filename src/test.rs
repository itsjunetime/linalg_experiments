#![cfg(test)]

use crate::{Matrix, Vector, matrix::InversionErr};

#[test]
fn mul_scalar() {
	let mx = Matrix::new([[1, 2], [3, 4]]);
	let tests = [
		(1, Matrix::new([[1, 2], [3, 4]])),
		(3, Matrix::new([[3, 6], [9, 12]])),
		(10, Matrix::new([[10, 20], [30, 40]]))
	];

	for (input, expected) in tests {
		assert_eq!(&mx * input, expected)
	}
}

#[test]
fn mul_matrix() {
	let mx = Matrix::new([[1, 2], [3, 4]]);
	assert_eq!(&mx * &Matrix::new([[1, 2], [3, 4]]), Matrix::new([[7, 10], [15, 22]]));
	assert_eq!(&mx * &Matrix::new([[1, 2, 3], [4, 5, 6]]), Matrix::new([[9, 12, 15], [19, 26, 33]]));
	assert_eq!(
		&Matrix::new([[1, 2]]) * &Matrix::new([[3], [4]]),
		Matrix::new([[11]])
	);
}

#[test]
fn add_vec() {
	let v = Vector::new([[1, 2, 3]]);
	let tests = [
		(Vector::new([[4, 5, 6]]), Vector::new([[5, 7, 9]])),
		(Vector::new([[0, 0, 0]]), Vector::new([[1, 2, 3]])),
		(Vector::new([[10, 9, 8]]), Vector::new([[11, 11, 11]]))
	];

	for (input, expected) in tests {
		assert_eq!(&v + &input, expected);
	}
}

#[test]
fn sub_vec() {
	let v = Vector::new([[4, 5, 6]]);
	let tests = [
		(Vector::new([[4, 5, 6]]), Vector::new([[0, 0, 0]])),
		(Vector::new([[0, 0, 0]]), Vector::new([[4, 5, 6]])),
		(Vector::new([[3, 2, 1]]), Vector::new([[1, 3, 5]]))
	];

	for (input, expected) in tests {
		assert_eq!(&v - &input, expected);
	}
}

#[test]
fn matrix_determinant() {
	// just make boilerplate easier
	macro_rules! eq{ ($arr:expr, $val:expr) => {
		assert_eq!(Matrix::new($arr).determinant(), $val);
	}}

	// stole these examples from https://www.symbolab.com/solver/matrix-determinant-calculator
	eq!([[1.]], 1.);
	eq!([[1., 2.], [3., 4.]], -2.0);
	eq!([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]], 0.);
	eq!([[1., 3., 5., 9.], [1., 3., 1., 7.], [4., 3., 9., 7.], [5., 2., 0., 9.]], -376.);
}

#[test]
fn matrix_inverse() {
	// just make boilerplate easier
	macro_rules! eq{ ($arr:expr, $val:expr) => {
		assert_eq!(Matrix::new($arr).inverse(), Ok(Matrix::new($val)));
	}}

	// also stole these from https://www.symbolab.com/solver/matrix-inverse-calculator but it says
	// that the third one should have a valid inverse, but its determinant is also 0, so I don't
	// know how they got that. The 'valid' inverse is commented out.
	eq!([[1.]], [[1.]]);
	eq!([[1., 2.], [3., 4.]], [[-2., 1.], [1.5, -0.5]]);
	assert_eq!(
		Matrix::new([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]).inverse(),
		Err(InversionErr::DeterminantIsZero)
		// [[-11./12., 1./3., 1./12.], [-1./6., 1./3., -1./6.], [0.75, -1./3., 1./12.]]
	);
	eq!(
		[[1., 3., 5., 9.], [1., 3., 1., 7.], [4., 3., 9., 7.], [5., 2., 0., 9.]], 
		[
			[-13./47., 2./47., 7./47., 6./47.],
			[-0.625, 0.875, 0.25, -0.25],
			[39./376., -53./376., 13./188., -9./188.], 
			[55./188., -41./188., -13./94., 9./94.]
		]
	);
}

#[test]
fn vector_dot_product() {
	let base = Vector::new([[1., 2., 3.]]);
	let tests = [
		(Vector::new([[10., 10., 10.]]), 60.),
		(Vector::new([[0., -1., 1.]]), 1.),
		(Vector::new([[12., 21., 149.]]), 501.)
	];

	for (input, expected) in tests {
		assert_eq!(input.dot_product(&base), expected);
	}
}
