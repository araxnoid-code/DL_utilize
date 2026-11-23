use ndarray::{Array1, Array2, Array3, Array4, Array5, Array6, ArrayD};
use num_traits::Float;

// MARK
pub trait NDArrayMark {
    type F: Float;
}
impl<F> NDArrayMark for Array1<F>
where
    F: Float,
{
    type F = F;
}

impl<F> NDArrayMark for Array2<F>
where
    F: Float,
{
    type F = F;
}

impl<F> NDArrayMark for Array3<F>
where
    F: Float,
{
    type F = F;
}

impl<F> NDArrayMark for Array4<F>
where
    F: Float,
{
    type F = F;
}

impl<F> NDArrayMark for Array5<F>
where
    F: Float,
{
    type F = F;
}

impl<F> NDArrayMark for Array6<F>
where
    F: Float,
{
    type F = F;
}

impl<F> NDArrayMark for ArrayD<F>
where
    F: Float,
{
    type F = F;
}

// NEG
pub trait RefNumNeg {
    type Out;
    fn neg(&self) -> Self::Out;
}

impl<'a, F> RefNumNeg for &'a F
where
    F: Float,
{
    type Out = F;
    fn neg(&self) -> Self::Out {
        -**self
    }
}
