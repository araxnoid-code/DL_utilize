// # MulTool
pub struct MulTool<'a, Lhs, Rhs> {
    pub lhs: &'a Lhs,
    pub rhs: &'a Rhs,
}

impl<'a, Lhs, Rhs> MulTool<'a, Lhs, Rhs> {
    pub fn new(lhs: &'a Lhs, rhs: &'a Rhs) -> MulTool<'a, Lhs, Rhs> {
        Self { lhs, rhs }
    }
}

// # MulFlow
pub struct MulFlow {
    pub value_idx: usize,
    pub grad_idx: Option<usize>,
    //
    pub lhs_idx: usize,
    pub lhs_grad_idx: Option<usize>,
    pub rhs_idx: usize,
    pub rhs_grad_idx: Option<usize>,
}
