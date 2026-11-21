use ndarray::{Array1, Array2, Array3, Array4, Array5, Array6, ArrayD};

pub trait NDArrayMark {}
impl<F> NDArrayMark for Array1<F> {}
impl<F> NDArrayMark for Array2<F> {}
impl<F> NDArrayMark for Array3<F> {}
impl<F> NDArrayMark for Array4<F> {}
impl<F> NDArrayMark for Array5<F> {}
impl<F> NDArrayMark for Array6<F> {}
impl<F> NDArrayMark for ArrayD<F> {}
