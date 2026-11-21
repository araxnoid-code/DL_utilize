use crate::AddFlow;

#[derive(Clone, Copy)]
pub enum Label<Array, Ref> {
    Add(AddFlow<Array, Ref>),
}

pub trait CopyLabel {
    fn copy_label(&self) -> Self;
}

impl<Array, Ref> CopyLabel for Label<Array, Ref> {
    fn copy_label(&self) -> Self {
        match self {
            Self::Add(add_flow) => Label::Add(add_flow._copy()),
        }
    }
}
