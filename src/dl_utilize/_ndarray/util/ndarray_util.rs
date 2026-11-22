use ndarray::{Array1, Array2, Array3, Array4, Array5, Array6, ArrayD};
use num_traits::Float;

// MARK
pub trait NDArrayMark {}
impl<F> NDArrayMark for Array1<F> {}
impl<F> NDArrayMark for Array2<F> {}
impl<F> NDArrayMark for Array3<F> {}
impl<F> NDArrayMark for Array4<F> {}
impl<F> NDArrayMark for Array5<F> {}
impl<F> NDArrayMark for Array6<F> {}
impl<F> NDArrayMark for ArrayD<F> {}

// NEG
pub trait RefNumNeg {
    type Out;
    fn neg(&self) -> Self::Out;
}

// impl<F> RefNumNeg for F
// where
//     F: Float,
// {
//     type Out = F;
//     fn neg(&self) -> Self::Out {
//         -*self
//     }
// }

impl<'a, F> RefNumNeg for &'a F
where
    F: Float,
{
    type Out = F;
    fn neg(&self) -> Self::Out {
        -**self
    }
}

// impl<'a> RefNumNeg for &'a f32 {
//     type Out = f32;
//     fn neg(&self) -> Self::Out {
//         -*self
//     }
// }
//
// impl<'a> RefNumNeg for &'a f64 {
//     type Out = f64;
//     fn neg(&self) -> Self::Out {
//         -*self
//     }
// }
