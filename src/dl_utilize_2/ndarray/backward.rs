pub trait NDArrayBackward<Grad> {
    fn backward(&self, grads: &mut Vec<Option<Grad>>);
}
