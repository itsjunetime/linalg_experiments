use std::{
	ops::{Add, Mul, Sub, Neg, Div, Index, IndexMut},
	fmt::{Formatter, Debug, Display},
	mem::MaybeUninit
};
use crate::{ArrayCollectable, Assert, IsTrue, slice_chunks};

#[derive(Debug, PartialEq)]
/// A Matrix of size W x H (where W represents the width or number of columns and H represents the
/// height or number of rows)
pub struct Matrix<const W: usize, const H: usize, T> {
	pub(crate) inner: [[T; W]; H]
}

impl<const W: usize, const H: usize, T> Matrix<W, H, T> {
	/// Create a new matrix from the given array.
	///
	/// # Arguments
	///
	/// * `inner` - The array to store as the internal matrix representation
	///
	/// # Examples
	///
	/// ```compile_fail
	/// # use proj_4::prelude::Matrix;
	/// // Passing in an array of the wrong size will prevent compilation:
	/// let matrix: Matrix<2, 2, i32> = Matrix::new([[1]]);
	/// ```
	pub fn new(inner: [[T; W]; H]) -> Self {
		Matrix { inner }
	}
}

#[derive(Debug, PartialEq)]
pub enum InversionErr {
	DeterminantIsZero
}

impl<const S: usize, T> Matrix<S, S, T> {
	// Utility fn used both in determinant and inverse
	fn determinant_iter<'t>(slice: &[&[&'t T]]) -> T
	where
		&'t T: Mul<T, Output=T>,
		T: Add<Output=T> + Sub<Output=T> + Clone
	{
		let len = slice.len();
		if len == 1 {
			return slice[0][0].clone();
		}

		let mut iter = slice[0].iter()
			.enumerate()
			.map(|(col_idx, val)| {
				// We have to collect it into a vec so that we have a backing storage to get a
				// slice from
				let inner = slice[1..].iter().flat_map(|row|
					row.iter()
						.enumerate()
						.filter_map(|(idx, v)| (idx != col_idx).then_some(*v))
				).collect::<Vec<&T>>();

				// collect the slices into their own array to pass into the fn again
				let slice_vec = slice_chunks(&inner, len - 1);

				*val * Self::determinant_iter(slice_vec.as_slice())
			});

		// This is safe to unwrap 'cause we verify in the trait bounds that S > 0
		let first = iter.next().unwrap();

		iter.enumerate().fold(first, |acc, (idx, val)|
			if idx & 1 == 1 {
				acc + val
			} else {
				acc - val
			}
		)
	}

	/// Get the determinant of the matrix
	///
	/// # Examples
	///
	/// ```
	/// # #![feature(generic_const_exprs)]
	/// # use proj_4::prelude::Matrix;
	/// let mx = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
	/// assert_eq!(mx.determinant(), -2.0);
	/// ```
	pub fn determinant<'m>(&'m self) -> T
	where
		// We once again need the Output=T bound since this can get extremely nested, depending on the
		// size of the matrix, and that makes impossible recursive trait bounds
		// Also we unfortunately need the T: Clone bound so that we can handle the case where S == 1
		// and we just need to clone the single value in the matrix and return that.
		&'m T: Mul<T, Output=T>,
		T: Clone + Sub<Output=T> + Add<Output=T>,
		// We can't find the determinant for a zero-sized matrix
		Assert<{ S > 0 }>: IsTrue
	{
		let ref_inner = self.inner
			.each_ref()
			.map(<[T; S]>::each_ref);

		let sliced: [&[&T]; S] = (0..S)
			.map(|idx| ref_inner[idx].as_slice())
			.collect_array()
			.unwrap();

		// ugh. Ok. This is a nightmare.
		// This inner function is required because we can't call determinant() on a matrix whose
		// dimensions are both 0 (what would we return?) but getting the determinant of a Matrix
		// size S requires us to get the determinant of Matrix size S - 1, and we have no way to
		// promise the compiler that we are specially handling S == 1 and thus a Matrix of size 0
		// will never be able to call this function. Because we can't express that, we need to have
		// a function that works on a slice/array/non-matrix.
		// However, we also ran into some issues here. basically, const generic expressions are
		// experimental for a reason. The compiler seems to choke if I make this function generic
		// over a const, and then call e.g. determinant_iter::<{S-1}, T>(...) to handle the
		// recursive cases. So I need to make it take a slice so that the compiler doesn't have to
		// typecheck const generics and instead just trusts me that everything is the right size.
		Self::determinant_iter(sliced.as_slice())
	}

	/// Get the inverse of the given matrix. This requires that the matrix is a square, and will
	/// return an error if the determinant is a value that we can't divide by (e.g. 0)
	///
	/// # Examples
	///
	/// ```
	/// # #![feature(generic_const_exprs)]
	/// # use proj_4::prelude::Matrix;
	/// let mx = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
	/// let inverse = Matrix::new([[-2.0, 1.0], [1.5, -0.5]]);
	/// assert_eq!(mx.inverse().unwrap(), inverse);
	/// ```
	pub fn inverse<'m>(&'m self) -> Result<Self, InversionErr>
	where
		&'m T: Mul<T, Output=T> + Div<&'m T, Output=T>,
		// We add the Default bound here as a kind of cheating way to detect if we're going to
		// divide by 0 while still being generic 
		T: Clone + Sub<Output=T> + Add<Output=T> + Neg<Output=T> + Default + PartialEq,
		for<'i> T: Div<&'i T, Output=T>,
		// We can't find the determinant for a zero-sized matrix
		Assert<{ S > 0 }>: IsTrue
	{
		// We need to handle the edge case of S == 1 here or else we'll try to get the cofactor
		// elements for it, but those don't exist, and we'll just run into a lot of issues later.
		// Also we can't just return a Matrix of size 1x1, we have to return a matrix of size SxS,
		// so we have to do some tricks to let type inference do its thing even though we know it's
		// just 1x1
		if S == 1 {
			// SAFETY: This is safe because we're creating a structure filled with uninits, so it's
			// fine that it's actually not initialized
			let mut inner_arr: [[MaybeUninit<T>; S]; S] = unsafe { MaybeUninit::uninit().assume_init() };
			let single_val = &self.inner[0][0];
			// We just get the inverse of the single value in the 1x1 matrix; we don't even need
			// the determinant
			// Yes, clippy, I know this basically just resolves to 1.0 but we can't say that 'cause
			// we don't know what type they're using
			#[allow(clippy::eq_op)]
			inner_arr[0][0].write((single_val / single_val) / single_val);
			// SAFETY: We've ensured that S == 1, so this is an array like [[T; 1]; 1] and thus
			// [0][0] is the only index we need to write to to be able to unwrap each one
			return Ok(Matrix::new(inner_arr.map(|a| a.map(|m| unsafe { m.assume_init() }))))
		}

		let det = self.determinant();
		if det == T::default() {
			return Err(InversionErr::DeterminantIsZero)
		}

		// this is already transposed, so we can skip that step
		Ok(Matrix::new((0..S).map(|og_col_idx| {
			(0..S).map(|og_row_idx| {
				// for each item, get the values that we need to create the cofactor by only
				// grabbing the items not in its row/column (but inverting row/column matching to
				// automatically transpose the generated matrix)
				let cofactor_vals = self.inner.iter()
					.enumerate()
					.filter(|(row_idx, _)| *row_idx != og_row_idx)
					.flat_map(|(_, row)| {
						row.iter()
							.enumerate()
							.filter_map(|(col_idx, val)| (col_idx != og_col_idx).then_some(val))
					})
					.collect::<Vec<&T>>();

				// Do some vector slicing to put the values into an easier-to-work-with container
				let cofactor_vec = slice_chunks(&cofactor_vals, S - 1);

				// And get the cofactor (which is calculated basically the same as the determinant)
				// for that value.
				// We divide each cofactor by the determinant here to essentially multiply the whole
				// matrix by 1 / determinant
				let cofactor_div = Self::determinant_iter(cofactor_vec.as_slice()) / &det;
				if (og_col_idx + og_row_idx) & 1 == 1 {
					// Negate it 'cause I guess that's what you do
					-cofactor_div
				} else {
					cofactor_div
				}
			})
			// This is safe to unwrap 'cause (0..S).len() == S 
			.collect_array()
			.unwrap()
		})
		// This is safe to unwrap 'cause (0..S).len() == S 
		.collect_array()
		.unwrap()))
	}
}

/// Print a pretty depiction of an array
///
/// # Examples
///
/// ```
/// # use proj_4::prelude::Matrix;
/// let expected = String::from("┌ 1 2 ┐\n└ 3 4 ┘");
/// assert_eq!(format!("{}", Matrix::new([[1, 2], [3, 4]])), expected);
/// // Prints:
/// // ┌ 1 2 ┐
/// // └ 3 4 ┘
/// ```
impl<const W: usize, const H: usize, T> Display for Matrix<W, H, T> where T: Display + Debug {
	fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
		self.inner.iter().enumerate().try_for_each(|(row_idx, row)| {
			let (start, end) = match row_idx {
				0 => ("┌ ", " ┐\n"),
				i if i == H - 1 => ("└ ", " ┘"),
				_ => ("│ ", " │\n")
			};
			write!(f, "{start}")?;

			row.iter().enumerate().try_for_each(|(idx, val)| {
				std::fmt::Display::fmt(&val, f)?;
				if idx != W - 1 {
					write!(f, " ")?;
				}
				Ok(())
			})?;
			write!(f, "{end}")
		})
	}
}

/// Multiply a matrix by a scalar value, taking both by reference
///
/// # Examples
///
/// ```
/// # use proj_4::prelude::Matrix;
/// let mx = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
/// let mul_val = 4.0;
/// let new_mx = &mx * &mul_val;
/// ```
impl<'m, 't, const W: usize, const H: usize, T> Mul<&'m T> for &'t Matrix<W, H, T>
where
	&'t T: Mul<&'m T>
{
	type Output = Matrix<W, H, <&'t T as Mul<&'m T>>::Output>;
	fn mul(self, other: &'m T) -> Self::Output {
		Matrix {
			inner: self.inner.each_ref()
				.map(|row| row.each_ref().map(|val| val * other))
		}
	}
}

/// Multiply a matrix by a scalar value, consuming the matrix but only taking the scalar value by
/// reference.
///
/// # Examples
///
/// ```
/// # use proj_4::prelude::Matrix;
/// let mx = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
/// let mul_val = 4.0;
/// let new_mx = mx * &mul_val;
/// ```
impl<'m, const W: usize, const H: usize, T> Mul<&'m T> for Matrix<W, H, T>
where
	T: Mul<&'m T>
{
	type Output = Matrix<W, H, <T as Mul<&'m T>>::Output>;
	fn mul(self, other: &'m T) -> Self::Output {
		Matrix {
			inner: self.inner.map(|row| row.map(|val| val * other))
		}
	}
}

/// Multiply a matrix by a scalar value, consuming only the scalar value and taking the matrix by
/// reference.
///
/// # Examples
///
/// ```
/// # use proj_4::prelude::Matrix;
/// let mx = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
/// let new_mx = &mx * 4.0;
/// ```
impl<'t, const W: usize, const H: usize, T> Mul<T> for &'t Matrix<W, H, T>
where
	// So. The Output=T bound is required here so that we can actually state the type Output. If we
	// didn't do Output=T, Output would have to be Matrix<W, H, <&'t T as Mul<&'??? T>>::Output>
	// see, the issue is that we can't specify the lifetime of the other &T. If we introduce
	// another lifetime in the impl<>, the compiler will error because it's not constrained. We
	// can't do an anonymous lifetime, we can't do for<'i>, we cant say type Output<'a> 'cause
	// that's not in the trait definition, etc. annoying.
	for<'i> &'t T: Mul<&'i T, Output=T>,
{
	type Output = Matrix<W, H, T>;
	fn mul(self, other: T) -> Self::Output {
		// in the body here, we have to reimplement the multipliciation and can't just do
		// <Self as Mul<&T>>::mul(self, &other) because the impl Mul<&T> doesn't constrain that
		// <T as Mul<&T>>::Output == T, so it wouldn't match our type signature
		Matrix {
			inner: self.inner.each_ref()
				.map(|row| row.each_ref().map(|val| val * &other))
		}
	}
}

/// Multiply a matrix by a scalar value, consuming both.
///
/// # Examples
///
/// ```
/// # use proj_4::prelude::Matrix;
/// let mx = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
/// let new_mx = mx * 4.0;
/// ```
impl<const W: usize, const H: usize, T> Mul<T> for Matrix<W, H, T>
where
	for<'i> T: Mul<&'i T, Output=T>,
{
	type Output = Matrix<W, H, T>;
	fn mul(self, other: T) -> Self::Output {
		Matrix {
			inner: self.inner.map(|row| row.map(|val| val * &other))
		}
	}
}

/// Multiply two matrices. Their dimensions must be compatible (the width of self must be equal to
/// the height of the other), and this operation consumes neither but produces a completely new
/// matrix
///
/// # Examples
///
/// ```compile_fail
/// # #![feature(generic_const_exprs)]
/// # use proj_4::prelude::Matrix;
/// // These matrix dimensions are not compatible
/// let m1: Matrix<1, 2, i32> = Matrix::new([[1], [2]]);
/// let m2: Matrix<3, 3, i32> = Matrix::new([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
/// let new_matrix = &m1 * &m2;
/// ```
///
/// ```
/// # #![feature(generic_const_exprs)]
/// # use proj_4::prelude::Matrix;
/// // The matrix dimensions *are* compatible
/// let m1: Matrix<1, 2, i32> = Matrix::new([[1], [2]]);
/// let m2: Matrix<3, 1, i32> = Matrix::new([[1, 2, 3]]);
/// let new_matrix = &m1 * &m2;
/// ```
impl<'m1, 'm2, const W: usize, const H: usize, const W2: usize, T> Mul<&'m1 Matrix<W2, W, T>> for &'m2 Matrix<W, H, T>
where
	// We have to constrain Output=T here for the reason of multiple addition. In the body, we have
	// to sum up all the multiplied values with fold, and that requires the value returned in the
	// fold body to be the same every time. However, we can't constrain here that:
	// <&'m2 T as Mul<&'m1 T>>::Output: Add,
	// <&'m2 T as Mul<&'m1 T>::Output == <<&'m2 T as Mul<&'m1 T>>::Output as Add>::Output
	// if we can't constrain that, each call to the fold body would be a new type (as it would go
	// deeper into the <<<X as Add>::Output as Add>::Output as Add>::Output, etc, hole).
	&'m2 T: Mul<&'m1 T, Output = T>,
	T: Add<Output = T>,
	// You shouldn't be able to multiply zero-sized matrices, I think. That way, we can remove the
	// T: Default bound since that's required if we can't reliably unwrap the first element in a matrix
	Assert<{ W > 0 }>: IsTrue,
	Assert<{ H > 0 }>: IsTrue
{
	type Output = Matrix<W2, H, T>;
	fn mul(self, other: &'m1 Matrix<W2, W, T>) -> Self::Output {
		Matrix {
			// iterate through all the rows
			inner: self.inner
				.each_ref()
				// for each row...
				.map(|row|
					// iterate over each column in `other`
					(0..W2).map(|c_idx| {
						// multiply all the elements from this row and from `other`'s column
						let mut iter = row.iter().enumerate().map(|(idx, val)| val * &other.inner[idx][c_idx]);
						// This is safe to unwrap because the const generics verify that H > 0 &&
						// W > 0, and row (which we are iterating over) is len of H.
						let first = iter.next().unwrap();
						// trust me clippy, I'm supposed to add here
						#[allow(clippy::suspicious_arithmetic_impl)]
						iter.fold(first, |sum, val| sum + val)
					})
					.collect_array()
					// this is safe to unwrap because (0..H).len() == H
					.unwrap()
				)
		}
	}
}

impl<T> From<(T, T, T, T)> for Matrix<2, 2, T> {
	/// Format the tuple into a Matrix where the elements are arranged in a 2x2 layout, such that
	/// it looks like [[a, b], [c, d]]
	///
	/// # Examples
	///
	/// ```
	/// # use proj_4::prelude::Matrix;
	/// let mx: Matrix<2, 2, f32> = (1., 2., 3., 4.).into();
	/// ```
	fn from((a, b, c, d): (T, T, T, T)) -> Self {
		Matrix { inner: [[a, b], [c, d]] }
	}
}

/// Index into the backing array for easy access, iteration, and manipulation
impl<const W: usize, const H: usize, T, I> Index<I> for Matrix<W, H, T> where [[T; W]; H]: Index<I> {
	type Output = <[[T; W]; H] as Index<I>>::Output;

	fn index(&self, idx: I) -> &Self::Output {
		&self.inner[idx]
	}
}

/// Index into the backing array for easy access, iteration, and manipulation
impl<const W: usize, const H: usize, T, I> IndexMut<I> for Matrix<W, H, T> where [[T; W]; H]: IndexMut<I> {
	fn index_mut(&mut self, idx: I) -> &mut Self::Output {
		&mut self.inner[idx]
	}
}

/// Convert Matrix into an iterator over its inner array
impl<const W: usize, const H: usize, T> IntoIterator for Matrix<W, H, T> {
	type Item = <[[T; W]; H] as IntoIterator>::Item;
	type IntoIter = <[[T; W]; H] as IntoIterator>::IntoIter;
	fn into_iter(self) -> Self::IntoIter {
		self.inner.into_iter()
	}
}

/// Convert borrowed Matrix into an iterator over references of its inner arrays
impl<'m, const W: usize, const H: usize, T> IntoIterator for &'m Matrix<W, H, T> {
	type Item = <&'m [[T; W]; H] as IntoIterator>::Item;
	type IntoIter = <&'m [[T; W]; H] as IntoIterator>::IntoIter;
	fn into_iter(self) -> Self::IntoIter {
		// The type signature will be wrong if I just call iter()
		#[allow(clippy::into_iter_on_ref)]
		(&self.inner).into_iter()
	}
}

/// Convert a borrowed mutable Matrix into an iterator over mutable references of its inner arrays
impl<'m, const W: usize, const H: usize, T> IntoIterator for &'m mut Matrix<W, H, T> {
	type Item = <&'m mut [[T; W]; H] as IntoIterator>::Item;
	type IntoIter = <&'m mut [[T; W]; H] as IntoIterator>::IntoIter;
	fn into_iter(self) -> Self::IntoIter {
		// The type signature will be wrong if I just call iter_mut()
		#[allow(clippy::into_iter_on_ref)]
		(&mut self.inner).into_iter()
	}
}
