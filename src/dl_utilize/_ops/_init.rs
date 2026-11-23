pub trait InitTool<V, R, Flow> {
    fn push(self, values: &mut Vec<V>, refs: &mut Vec<R>) -> Flow;
}

pub struct InitFlow {
    pub value_idx: usize,
}

pub struct InitViewFlow {
    pub value_idx: usize,
}
