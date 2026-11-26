pub struct MulTool<'a, LI, RI, Own> {
    pub lhs: &'a LI,
    pub lhs_grad_idx: Option<usize>,
    pub rhs: &'a RI,
    pub rhs_grad_idx: Option<usize>,
    pub value: Own,
    pub grad_idx: Option<usize>,
}
