use std::ops::AddAssign;

use ndarray::{Dimension, IntoNdProducer};
use num_traits::Float;

use crate::{Gradient, Label, NDArrayMark};

pub trait NDArrayLabelBackward<Array, Grad, Dim>
where
    Array: NDArrayMark,
{
    fn backward(&self, values: &Vec<Array>, gradients: &mut Vec<Option<Grad>>);
}

impl<Array, Grad, Dim> NDArrayLabelBackward<Array, Grad, Dim> for Label
where
    for<'a> &'a mut Grad: IntoNdProducer<Dim = Dim, Item = &'a mut Grad::F>,
    for<'a> &'a Grad: IntoNdProducer<Dim = Dim, Item = &'a Grad::F>,
    for<'a> Grad: AddAssign<&'a Grad>,
    Grad: AddAssign<Grad::F> + Gradient,
    Grad::F: Float + AddAssign<Grad::F>,
    Array: NDArrayMark,
    Dim: Dimension,
{
    fn backward(&self, _: &Vec<Array>, gradients: &mut Vec<Option<Grad>>) {
        match self {
            Label::Add(add_flow) => {
                add_flow.backward::<_>(gradients);
            }
            Label::Sub(sub_flow) => {
                sub_flow.backward::<Grad, Dim>(gradients);
            }
        }
    }
}
