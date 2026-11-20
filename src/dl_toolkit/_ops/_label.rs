use crate::AddFlow;

#[derive(Clone, Copy)]
pub enum Label<F> {
    Add(AddFlow<F>),
}
