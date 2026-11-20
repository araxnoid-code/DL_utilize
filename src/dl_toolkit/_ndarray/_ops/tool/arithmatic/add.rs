use crate::{AddFlow, AddTool, GetValue, Label, ToolComponent};
use ndarray::ArrayD;
use std::ops::Add;

impl<F> GetValue for AddFlow<F> {
    type ValueType = ArrayD<F>;
    type ReturnType = ArrayD<F>;

    fn get_value<'a>(&'a self, values: &'a Vec<Self::ValueType>) -> &'a Self::ReturnType {
        &values[self.value_idx]
    }

    fn get_value_idx(&self) -> usize {
        self.value_idx
    }
}

impl<'t, Lhs, Rhs, F> ToolComponent<F, ArrayD<F>, AddFlow<F>> for AddTool<'t, Lhs, Rhs>
where
    Lhs: GetValue<ValueType = ArrayD<F>>,
    Rhs: GetValue<ValueType = ArrayD<F>>,
    for<'a> &'a Lhs::ReturnType: Add<&'a Rhs::ReturnType, Output = ArrayD<F>>,
{
    fn tool_forward(&self, values: &Vec<ArrayD<F>>) -> ArrayD<F> {
        let lhs = self.lhs.get_value(values);
        let rhs = self.rhs.get_value(values);
        lhs.add(rhs)
    }

    fn shape(out: &ArrayD<F>) -> Vec<usize> {
        out.shape().to_vec()
    }

    fn into_flow(self, value_idx: usize, grad_idx: usize) -> Label<F> {
        Label::Add(AddFlow::new(
            self.lhs.get_value_idx(),
            self.rhs.get_value_idx(),
            value_idx,
            grad_idx,
        ))
    }

    fn copy_label_to_flow(label: Label<F>) -> AddFlow<F> {
        match label {
            Label::Add(add_flow) => add_flow,
        }
    }
}
