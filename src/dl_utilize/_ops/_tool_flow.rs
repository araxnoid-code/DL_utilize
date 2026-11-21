pub trait ToolFlow {
    type ValueType;
    type RefType;
    type ReturnType;

    fn get_value<'a>(
        &'a self,
        values: &'a Vec<Self::ValueType>,
        refs: &'a Vec<Self::RefType>,
    ) -> &'a Self::ReturnType;
    fn get_value_idx(&self) -> usize;
    fn get_gradient_idx(&self) -> Option<usize>;
}
