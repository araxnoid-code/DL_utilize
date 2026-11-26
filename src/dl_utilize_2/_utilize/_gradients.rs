pub trait Gradients {
    type F;
    type Out;
    fn zeros(shape: Vec<usize>) -> Self::Out;
    fn fill_ones(&mut self);
    fn fill(&mut self, x: Self::F);
}
