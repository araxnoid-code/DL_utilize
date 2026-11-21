use std::{marker::PhantomData, ops::AddAssign};

use crate::{Gradient, InitFlow, Label, NDArrayLabelBackward, NDArrayMark, ToolComponent, Toolkit};

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
