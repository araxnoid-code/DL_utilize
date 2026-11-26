pub struct SubTool<'a, LI, RI, Own> {
    pub lhs: &'a LI,
    pub lhs_grad: Option<usize>,
    pub rhs: &'a RI,
    pub rhs_grad: Option<usize>,
    pub value: Own,
    pub grad_idx: Option<usize>,
}
