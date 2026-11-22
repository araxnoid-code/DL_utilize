use std::{marker::PhantomData, ops::AddAssign};

use num_traits::{Float, One};

use crate::{
    Gradient, InitFlow, Label, NDArrayLabelBackward, NDArrayMark, ToolComponent, ToolFlow, Toolkit,
};

impl<Array, Grad, Ref> Toolkit<Array, Grad, Ref, Label> {
    pub fn new() -> Toolkit<Array, Grad, Ref, Label> {
        Self {
            values: Vec::new(),
            gradients: Vec::new(),
            refs: Vec::new(),
            flows: Vec::new(),
        }
    }

    pub fn backward_from<Flow>(&mut self, flow: &Flow)
    where
        Array: NDArrayMark,
        Flow: ToolFlow<Array, Ref>,
        for<'a> Grad: Gradient + AddAssign<&'a Grad>,
        Grad::F: Float,
    {
        self.grad_into_ones(flow);
        self.backward();
    }

    pub fn backward(&mut self)
    where
        Array: NDArrayMark,
        Label: NDArrayLabelBackward<Array, Grad>,
        for<'a> Grad: AddAssign<&'a Grad>,
    {
        for label in self.flows.iter().rev() {
            label.backward(&self.values, &mut self.gradients);
        }
    }
}
