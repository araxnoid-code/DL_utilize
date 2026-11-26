use crate::Gradients;
use ndarray::{Array, ArrayD, ArrayRef};
use num_traits::{Float, Zero};

impl<F> Gradients for ArrayD<F>
where
    F: Clone + Float,
{
    type Out = ArrayD<F>;
    fn zeros(shape: Vec<usize>) -> Self::Out {
        ArrayD::zeros(shape)
    }

    fn fill_ones(&mut self) {
        ArrayRef::fill(self, F::one());
    }
}
