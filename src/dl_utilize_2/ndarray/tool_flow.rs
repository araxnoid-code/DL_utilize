use crate::ToolFLow;
use ndarray::{ArrayD, ArrayViewD};

impl<'v, F> ToolFLow for ArrayViewD<'v, F> {
    type Own = ArrayViewD<'v, F>;

    fn get_value<'a>(&'a self) -> &'a ArrayViewD<'v, F> {
        self
    }
    fn get_grad_idx(&self) -> Option<usize> {
        None
    }
}

impl<F> ToolFLow for ArrayD<F> {
    type Own = ArrayD<F>;

    fn get_value<'a>(&'a self) -> &'a ArrayD<F> {
        self
    }
    fn get_grad_idx(&self) -> Option<usize> {
        None
    }
}
