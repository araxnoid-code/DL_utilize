use crate::Label;

pub trait ToolComponent<F, Out, FlowType> {
    fn tool_forward(&self, values: &Vec<Out>) -> Out;
    fn shape(out: &Out) -> Vec<usize>;
    fn into_flow(self, value_idx: usize, grad_idx: usize) -> Label<F>;
    fn copy_label_to_flow(label: Label<F>) -> FlowType;
}

pub trait GetValue {
    type ReturnType;
    type ValueType;

    fn get_value<'a>(&'a self, values: &'a Vec<Self::ValueType>) -> &'a Self::ReturnType;
    fn get_value_idx(&self) -> usize;
}
