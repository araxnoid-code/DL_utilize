pub trait Gradients {
    type Out;
    fn zeros(shape: Vec<usize>) -> Self::Out;
}
