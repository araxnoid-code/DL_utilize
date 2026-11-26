pub trait ToolFLow {
    type ValueType;

    fn get_value<'a>(&'a self) -> &'a Self::ValueType;
    fn get_grad_idx(&self) -> Option<usize>;
}
