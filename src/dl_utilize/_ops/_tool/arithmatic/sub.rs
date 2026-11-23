/// # SubTool
pub struct SubTool<'a, Lhs, Rhs> {
    pub lhs: &'a Lhs,
    pub rhs: &'a Rhs,
}

impl<'a, Lhs, Rhs> SubTool<'a, Lhs, Rhs> {
    pub fn new(lhs: &'a Lhs, rhs: &'a Rhs) -> SubTool<'a, Lhs, Rhs> {
        Self { lhs, rhs }
    }
}

/// # SubFlow
#[derive(Clone, Copy)]
pub struct SubFlow {
    pub lhs_idx: usize,
    pub lhs_grad_idx: Option<usize>,
    pub rhs_idx: usize,
    pub rhs_grad_idx: Option<usize>,
    //
    pub value_idx: usize,
    pub grad_idx: Option<usize>,
}
