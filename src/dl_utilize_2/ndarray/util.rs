use crate::NDArrayGetShape;
use ndarray::ArrayD;

impl<F> NDArrayGetShape for ArrayD<F> {
    fn shape(&self) -> Vec<usize> {
        self.shape().to_vec()
    }
}
