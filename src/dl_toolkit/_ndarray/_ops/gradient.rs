use crate::Gradient;
use ndarray::ArrayD;
use num_traits::Zero;

impl<F> Gradient for ArrayD<F>
where
    F: Clone + Zero,
{
    fn zeros(shape: Vec<usize>) -> Self {
        ArrayD::zeros(shape)
    }
}
