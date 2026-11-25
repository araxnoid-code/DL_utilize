pub trait InitTool<A, R, Flow> {
    fn push(self, values: &mut Vec<A>, refs: &mut Vec<R>) -> Flow;
}

pub struct InitFlow {
    pub value_idx: usize,
}

pub struct InitViewFlow {
    pub value_idx: usize,
}
