pub trait ToolComponent<Out, FlowType> {
    type LabelType;
    fn tool_forward(&self, values: &Vec<Out>) -> Out;
    fn shape(out: &Out) -> Vec<usize>;
    fn into_flow(self, value_idx: usize, grad_idx: Option<usize>) -> Self::LabelType;
    fn copy_label_to_flow(label: Self::LabelType) -> FlowType;
}
