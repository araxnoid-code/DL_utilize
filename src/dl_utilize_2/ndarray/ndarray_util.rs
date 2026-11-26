use num_traits::Float;

// NEG
pub trait RefNumNeg {
    type Out;
    fn neg(&self) -> Self::Out;
}

impl<'a, F> RefNumNeg for &'a F
where
    F: Float,
{
    type Out = F;
    fn neg(&self) -> Self::Out {
        -**self
    }
}
