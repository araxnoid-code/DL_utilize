pub trait Backward<Grad> {
    fn backward(&self, grads: &mut Vec<Option<Grad>>);
}
