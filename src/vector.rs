use crate::{ArrayCollectable, prelude::Matrix, Assert, IsTrue};
use std::ops::{Add, Sub, Mul};

pub type Vector<const S: usize, T> = Matrix<S, 1, T>;

impl<const S: usize, T> Vector<S, T> {
	/// Get the dot product of two vectors
	///
	/// # Examples
	///
	/// ```
	/// # #![feature(generic_const_exprs)]
	/// # use proj_4::prelude::Vector;
	/// let v1 = Vector::new([[1., 2., 3.]]);
	/// let v2 = Vector::new([[4., 5., 6.]]);
	/// assert_eq!(v1.dot_product(&v2), 32.);
	/// ```
	pub fn dot_product<'v, 'o>(&'v self, other: &'o Vector<S, T>) -> T
	where
		&'v T: Mul<&'o T, Output=T>,
		T: Add<Output=T>,
		// we add this bound so that we don't have to bound T: Default
		Assert<{ S > 0 }>: IsTrue
	{
		let mut iter = self.inner[0].iter()
			.zip(other.inner[0].iter())
			.map(|(o, s)| o * s);
		// This is safe to unwrap because we ensure at compile time that S > 0, so this iterator
		// has to have at least 1 item
		let first = iter.next().unwrap();
		iter.fold(first, |acc, val| acc + val)
	}
}

impl<T> From<(T, T)> for Vector<2, T> {
	/// Format the tuple into a Vector, with the two elements being the only ones present in the
	/// vector.
	///
	/// # Examples
	///
	/// ```
	/// # use proj_4::prelude::Vector;
	/// let vec: Vector<2, f32> = (1., 2.).into();
	/// ```
	fn from((x, y): (T, T)) -> Self {
		Vector { inner: [[x, y]] }
	}
}

// Yes I could make consuming add and sub for vector as well so that it plays friendly but I'm
// tired of trait bounds and type definitions

/// Add one vector to another, consuming neither and producing a completely new vector
///
/// # Examples
///
/// ```compile_fail
/// # use proj_4::prelude::Vector;
/// // You cannot add two vectors of different sizes
/// let v1: Vector<1, i32> = Vector::new([[1]]);
/// let v2: Vector<2, i32> = Vector::new([[1, 2]]);
/// let new_vector = &v1 + &v2;
/// ```
impl<'v, 'a, const S: usize, T> Add<&'a Vector<S, T>> for &'v Vector<S, T> where &'v T: Add<&'a T> {
	type Output = Vector<S, <&'v T as Add<&'a T>>::Output>;
	fn add(self, other: &'a Vector<S, T>) -> Self::Output {
		Vector {
			inner: [self.inner[0].iter()
				.zip(other.inner[0].iter())
				.map(|(a, b)| a + b)
				.collect_array()
				.unwrap()]
		}
	}
}

/// Subtract one vector from another, consuming neither and producing a completely new vector
///
/// # Examples
///
/// ```compile_fail
/// # use proj_4::prelude::Vector;
/// // You cannot subtract two vectors of different sizes
/// let v1: Vector<1, i32> = Vector::new([[1]]);
/// let v2: Vector<2, i32> = Vector::new([[1, 2]]);
/// let new_vector = &v1 - &v2;
/// ```
impl<'v, 's, const S: usize, T> Sub<&'s Vector<S, T>> for &'v Vector<S, T> where &'v T: Sub<&'s T> {
	type Output = Vector<S, <&'v T as Sub<&'s T>>::Output>;
	fn sub(self, other: &'s Vector<S, T>) -> Self::Output {
		Vector {
			inner: [self.inner[0].iter()
				.zip(other.inner[0].iter())
				.map(|(a, b)| a - b)
				.collect_array()
				.unwrap()]
		}
	}
}
