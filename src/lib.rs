#![feature(array_methods)]
#![feature(generic_const_exprs)]

use std::mem::MaybeUninit;

mod test;
mod matrix;
mod vector;

pub mod prelude {
	pub use crate::{matrix::Matrix, vector::Vector};
}

// to not break semver
pub use matrix::Matrix;
pub use vector::Vector;

// To conform with the project requirements
pub type SimpleMatrix = Matrix<2, 2, f32>;
pub type SimpleVector = Vector<2, f32>;

trait ArrayCollectable<const S: usize, T> {
	// Returns the array if it turned out that self.len() == S, otherwise returns Err(self.len())
	fn collect_array(self) -> Result<[T; S], usize>;
}

impl<const S: usize, T, I> ArrayCollectable<S, T> for I where I: Iterator<Item = T> {
	fn collect_array(self) -> Result<[T; S], usize> {
		// SAFETY: This is safe because we are initializing a bunch of MaybeUninits, which are
		// expected to not be fully initialized, so them being in a uninitialized state is fine.
		// https://doc.rust-lang.org/core/mem/union.MaybeUninit.html#initializing-an-array-element-by-element
		let mut array: [MaybeUninit<T>; S] = unsafe { MaybeUninit::uninit().assume_init() };

		let mut count = 0;
		for next in self {
			if count >= S {
				return Err(count)
			}
			array[count].write(next);
			count += 1;
		}

		if count != S {
			return Err(count)
		}

		// SAFETY: We must ensure that count == S to make sure that every single element in the
		// array was written to, and we do that in the line just before this
		Ok(array.map(|t| unsafe { t.assume_init() }))
	}
}

fn slice_chunks<T>(vec: &Vec<T>, size: usize) -> Vec<&[T]> {
	let len = vec.len();
	let mut idx = 0;
	let mut ret_vec = Vec::with_capacity(len / size);

	while idx < len {
		let old_idx = idx;
		idx += size;
		ret_vec.push(&vec[old_idx..idx]);
	}

	ret_vec
}

// We have to make these pub so that people can add them as bounds if they need to to write a
// function that calls some of our functions with these same bounds
pub enum Assert<const CHECK: bool> {}
pub trait IsTrue {}
impl IsTrue for Assert<true> {}
