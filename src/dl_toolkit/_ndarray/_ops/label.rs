use std::ops::AddAssign;

use crate::{Label, NDArrayMark};

pub trait NDArrayLabelBackward<Array, Grad>
where
    Array: NDArrayMark,
{
    fn backward(&self, values: &Vec<Array>, gradients: &mut Vec<Option<Grad>>);
}

impl<Array, Grad> NDArrayLabelBackward<Array, Grad> for Label<Array>
where
    Array: NDArrayMark,
    for<'a> Grad: AddAssign<&'a Grad>,
{
    fn backward(&self, values: &Vec<Array>, gradients: &mut Vec<Option<Grad>>) {
        match self {
            Label::Add(add_flow) => {
                add_flow.backward(gradients);
            }
        }
    }
}
