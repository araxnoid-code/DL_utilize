use std::marker::PhantomData;

pub struct InitFlow<F> {
    pub value_idx: usize,
    pub(crate) _phantom: PhantomData<F>,
}
