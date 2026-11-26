pub struct AddTool<'a, LhsI, RhsI, Own> {
    pub lhs: &'a LhsI,
    pub lhs_grad_idx: Option<usize>,
    pub rhs: &'a RhsI,
    pub rhs_grad_idx: Option<usize>,
    pub value: Own,
    pub grad_idx: Option<usize>,
}
