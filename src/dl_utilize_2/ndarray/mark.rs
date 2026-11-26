use ndarray::{ArrayD, ArrayViewD};

pub trait NDArrayMark {
    type OwnType;
}

impl<F> NDArrayMark for ArrayD<F> {
    type OwnType = ArrayD<F>;
}

impl<'a, F> NDArrayMark for ArrayViewD<'a, F> {
    type OwnType = ArrayD<F>;
}
