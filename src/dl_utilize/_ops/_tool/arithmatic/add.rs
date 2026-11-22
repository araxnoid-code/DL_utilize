use std::marker::PhantomData;

/// # AddTool
pub struct AddTool<'a, Lhs, Rhs> {
    pub lhs: &'a Lhs,
    pub rhs: &'a Rhs,
}

impl<'a, Lhs, Rhs> AddTool<'a, Lhs, Rhs> {
    pub fn new(lhs: &'a Lhs, rhs: &'a Rhs) -> AddTool<'a, Lhs, Rhs> {
        Self { lhs, rhs }
    }
}

/// # AddFlow
#[derive(Clone, Copy)]
pub struct AddFlow {
    pub lhs_idx: usize,
    pub lhs_grad_idx: Option<usize>,
    pub rhs_idx: usize,
    pub rhs_grad_idx: Option<usize>,

    pub value_idx: usize,
    pub grad_idx: Option<usize>,
    // pub(crate) _phantom: PhantomData<(Array, Ref)>,
}

impl AddFlow {
    pub fn new(
        lhs_idx: usize,
        lhs_grad_idx: Option<usize>,
        rhs_idx: usize,
        rhs_grad_idx: Option<usize>,
        value_idx: usize,
        grad_idx: Option<usize>,
    ) -> AddFlow {
        Self {
            lhs_idx,
            lhs_grad_idx,
            rhs_idx,
            rhs_grad_idx,
            value_idx,
            grad_idx,
            // _phantom: Default::default(),
        }
    }

    pub fn _copy(&self) -> AddFlow {
        AddFlow {
            value_idx: self.value_idx,

            lhs_idx: self.lhs_idx,
            lhs_grad_idx: self.lhs_grad_idx,

            rhs_idx: self.rhs_idx,
            rhs_grad_idx: self.rhs_grad_idx,

            grad_idx: self.grad_idx,
            // _phantom: Default::default(),
        }
    }
}
