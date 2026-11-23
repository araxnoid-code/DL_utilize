use crate::AddFlow;

#[derive(Clone, Copy)]
pub enum Label {
    Add(AddFlow),
}

pub trait CopyLabel {
    fn copy_label(&self) -> Self;
}

// impl CopyLabel for Label {
//     fn copy_label(&self) -> Self {
//         match self {
//             Self::Add(add_flow) => Label::Add(*add_flow),
//         }
//     }
// }
