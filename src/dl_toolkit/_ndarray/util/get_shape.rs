use ndarray::{Array1, Array2, Array3, Array4, Array5, Array6, ArrayD};

pub trait NDArrayGetShape {
    fn get_shape(&self) -> Vec<usize>;
}

impl<F> NDArrayGetShape for Array1<F> {
    fn get_shape(&self) -> Vec<usize> {
        self.shape().to_vec()
    }
}

impl<F> NDArrayGetShape for Array2<F> {
    fn get_shape(&self) -> Vec<usize> {
        self.shape().to_vec()
    }
}

impl<F> NDArrayGetShape for Array3<F> {
    fn get_shape(&self) -> Vec<usize> {
        self.shape().to_vec()
    }
}

impl<F> NDArrayGetShape for Array4<F> {
    fn get_shape(&self) -> Vec<usize> {
        self.shape().to_vec()
    }
}

impl<F> NDArrayGetShape for Array5<F> {
    fn get_shape(&self) -> Vec<usize> {
        self.shape().to_vec()
    }
}

impl<F> NDArrayGetShape for Array6<F> {
    fn get_shape(&self) -> Vec<usize> {
        self.shape().to_vec()
    }
}

impl<F> NDArrayGetShape for ArrayD<F> {
    fn get_shape(&self) -> Vec<usize> {
        self.shape().to_vec()
    }
}
