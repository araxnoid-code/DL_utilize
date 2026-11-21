use crate::{InitFlow, ToolFlow};

impl<Array> ToolFlow for InitFlow<Array> {
    type ValueType = Array;
    fn get_value<'a>(&'a self, values: &'a Vec<Self::ValueType>) -> &'a Self::ValueType {
        &values[self.value_idx]
    }

    fn get_value_idx(&self) -> usize {
        self.value_idx
    }

    fn get_gradient_idx(&self) -> Option<usize> {
        None
    }
}
