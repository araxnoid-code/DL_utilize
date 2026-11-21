use num_traits::Float;

pub trait Gradient {
    type F;
    fn zeros(shape: Vec<usize>) -> Self;
    fn fill(&mut self, value: Self::F);
}
