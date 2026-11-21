pub trait ToolFlow {
    type ValueType;

    fn get_value<'a>(&'a self, values: &'a Vec<Self::ValueType>) -> &'a Self::ValueType;
    fn get_value_idx(&self) -> usize;
    fn get_gradient_idx(&self) -> Option<usize>;
}
