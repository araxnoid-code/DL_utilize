pub trait ToolFlow<ArrayType, RefType> {
    type ReturnType;

    fn get_value<'a>(
        &'a self,
        values: &'a Vec<ArrayType>,
        refs: &'a Vec<RefType>,
    ) -> &'a Self::ReturnType;
    fn get_value_idx(&self) -> usize;
    fn get_gradient_idx(&self) -> Option<usize>;
}
