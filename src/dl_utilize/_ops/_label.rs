use crate::{AddFlow, SubFlow};

#[derive(Clone, Copy)]
pub enum Label {
    Add(AddFlow),
    Sub(SubFlow),
}

pub trait CopyLabel {
    fn copy_label(&self) -> Self;
}
