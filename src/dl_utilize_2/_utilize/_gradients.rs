pub trait Gradients {
    type Out;
    fn zeros(shape: Vec<usize>) -> Self::Out;
    fn fill_ones(&mut self);
}
