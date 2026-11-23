use std::ops::AddAssign;

use num_traits::Float;

use crate::{Gradient, Label, NDArrayLabelBackward, NDArrayMark, ToolFlow, Toolkit};

impl<Array, Grad, Ref> Toolkit<Array, Grad, Ref> {
    pub fn new() -> Toolkit<Array, Grad, Ref> {
        Self {
            values: Vec::new(),
            gradients: Vec::new(),
            refs: Vec::new(),
            label_flows: Vec::new(),
        }
    }

    pub fn backward_from<Flow, Dim>(&mut self, flow: &Flow)
    where
        Array: NDArrayMark,
        Flow: ToolFlow<Array, Ref>,
        for<'a> Grad: Gradient + AddAssign<&'a Grad>,
        Grad::F: Float,
        Label: NDArrayLabelBackward<Array, Grad, Dim>,
    {
        self.grad_into_ones(flow);
        self.backward::<Dim>();
    }

    pub fn backward<Dim>(&mut self)
    where
        Array: NDArrayMark,
        Label: NDArrayLabelBackward<Array, Grad, Dim>,
        for<'a> Grad: AddAssign<&'a Grad>,
    {
        for label in self.label_flows.iter().rev() {
            label.backward(&self.values, &mut self.gradients);
        }
    }
}
