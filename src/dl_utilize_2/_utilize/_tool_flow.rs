pub trait ToolFLow {
    type Own;

    fn get_value<'a>(&'a self) -> &'a Self::Own;
    fn get_grad_idx(&self) -> Option<usize>;
}
