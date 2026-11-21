use std::marker::PhantomData;

pub trait InitTool<V, R, Flow> {
    fn push(self, values: &mut Vec<V>, refs: &mut Vec<R>) -> Flow;
}

pub struct InitFlow<Array, Ref> {
    pub value_idx: usize,
    pub(crate) _phantom: PhantomData<(Array, Ref)>,
}

pub struct InitViewFlow<Array, Ref> {
    pub value_idx: usize,
    pub(crate) _phantom: PhantomData<(Array, Ref)>,
}
