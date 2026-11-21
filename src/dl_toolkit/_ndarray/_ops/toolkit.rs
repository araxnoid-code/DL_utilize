use std::{marker::PhantomData, ops::AddAssign};

use num_traits::{Float, One};

use crate::{
    Gradient, InitFlow, Label, NDArrayLabelBackward, NDArrayMark, ToolComponent, ToolFlow, Toolkit,
};

impl<Array, Grad> Toolkit<Array, Grad, Label<Array>> {
    pub fn new() -> Toolkit<Array, Grad, Label<Array>> {
        Self {
            values: Vec::new(),
            gradients: Vec::new(),
            flows: Vec::new(),
        }
    }

    pub fn init_value(&mut self, value: Array) -> InitFlow<Array> {
        self.values.push(value);
        InitFlow {
            value_idx: self.values.len() - 1,
            _phantom: Default::default(),
        }
    }

    pub fn backward_from<Flow>(&mut self, flow: &Flow)
    where
        Array: NDArrayMark,
        Flow: ToolFlow,
        for<'a> Grad: Gradient + AddAssign<&'a Grad>,
        Grad::F: Float,
    {
        self.grad_into_ones(flow);
        self.backward();
    }

    pub fn backward(&mut self)
    where
        Array: NDArrayMark,
        Label<Array>: NDArrayLabelBackward<Array, Grad>,
        for<'a> Grad: AddAssign<&'a Grad>,
    {
        for label in self.flows.iter().rev() {
            label.backward(&self.values, &mut self.gradients);
        }
    }
}
