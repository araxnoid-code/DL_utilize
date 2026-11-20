use std::marker::PhantomData;

pub struct AddTool<'a, Lhs, Rhs> {
    pub lhs: &'a Lhs,
    pub rhs: &'a Rhs,
}

impl<'a, Lhs, Rhs> AddTool<'a, Lhs, Rhs> {
    pub fn new(lhs: &'a Lhs, rhs: &'a Rhs) -> AddTool<'a, Lhs, Rhs> {
        Self { lhs, rhs }
    }
}

#[derive(Clone, Copy)]
pub struct AddFlow<F> {
    pub lhs_idx: usize,
    pub rhs_idx: usize,

    pub value_idx: usize,
    pub grad_idx: usize,
    pub(crate) _phantom: PhantomData<F>,
}

impl<F> AddFlow<F> {
    pub fn new(lhs_idx: usize, rhs_idx: usize, value_idx: usize, grad_idx: usize) -> AddFlow<F> {
        Self {
            lhs_idx,
            rhs_idx,
            value_idx,
            grad_idx,
            _phantom: Default::default(),
        }
    }
}
