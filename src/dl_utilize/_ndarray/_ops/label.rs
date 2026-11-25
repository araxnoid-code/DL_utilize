use std::ops::AddAssign;

use ndarray::{Dimension, IntoNdProducer};
use num_traits::Float;

use crate::{Gradient, Label, NDArrayMark};

pub trait NDArrayLabelBackward<A, R, G, Dim>
where
    A: NDArrayMark,
{
    fn backward(&self, values: &Vec<A>, refs: &Vec<R>, gradients: &mut Vec<Option<G>>);
}

impl<A, R, G, Dim> NDArrayLabelBackward<A, R, G, Dim> for Label
where
    for<'a> &'a mut G: IntoNdProducer<Dim = Dim, Item = &'a mut G::F>,
    for<'a> &'a G: IntoNdProducer<Dim = Dim, Item = &'a G::F>,
    for<'a> G: AddAssign<&'a G>,
    G: AddAssign<G::F> + Gradient,
    G::F: Float + AddAssign<G::F>,
    A: NDArrayMark,
    Dim: Dimension,
{
    fn backward(&self, values: &Vec<A>, refs: &Vec<R>, gradients: &mut Vec<Option<G>>) {
        match self {
            Label::Add(add_flow) => {
                add_flow.backward::<_>(gradients);
            }
            Label::Sub(sub_flow) => {
                sub_flow.backward::<G, Dim>(gradients);
            }
        }
    }
}
