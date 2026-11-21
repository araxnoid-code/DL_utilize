use crate::AddFlow;

#[derive(Clone, Copy)]
pub enum Label<Array> {
    Add(AddFlow<Array>),
}

pub trait CopyLabel {
    fn copy_label(&self) -> Self;
}

impl<Array> CopyLabel for Label<Array> {
    fn copy_label(&self) -> Self {
        match self {
            Self::Add(add_flow) => Label::Add(add_flow._copy()),
        }
    }
}
