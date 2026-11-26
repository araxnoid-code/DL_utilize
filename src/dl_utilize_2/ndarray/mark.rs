use ndarray::{ArrayD, ArrayViewD};

pub trait NDArrayMark {
    type OwnType;
    type F;
}

impl<F> NDArrayMark for ArrayD<F> {
    type OwnType = ArrayD<F>;
    type F = F;
}

impl<'a, F> NDArrayMark for ArrayViewD<'a, F> {
    type OwnType = ArrayD<F>;
    type F = F;
}
