use ndarray::ArrayD;

use crate::{GetValue, InitFlow};

impl<F> GetValue for InitFlow<F> {
    type ReturnType = ArrayD<F>;
    type ValueType = ArrayD<F>;
    fn get_value<'a>(&'a self, values: &'a Vec<Self::ValueType>) -> &'a Self::ReturnType {
        &values[self.value_idx]
    }

    fn get_value_idx(&self) -> usize {
        self.value_idx
    }
}
