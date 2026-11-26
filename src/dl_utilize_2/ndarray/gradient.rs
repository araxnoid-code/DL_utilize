use crate::Gradients;
use ndarray::ArrayD;
use num_traits::Zero;

impl<F> Gradients for ArrayD<F>
where
    F: Clone + Zero,
{
    type Out = ArrayD<F>;
    fn zeros(shape: Vec<usize>) -> Self::Out {
        ArrayD::zeros(shape)
    }
}
