use crate::{GetValue, Label};

pub struct Toolkit<F, V, Grad> {
    pub(crate) values: Vec<V>,
    pub(crate) gradients: Vec<Grad>,
    pub(crate) flows: Vec<Label<F>>,
}

impl<F, V, Grad> Toolkit<F, V, Grad> {
    pub fn get_value<Flow>(&self, flow: &Flow) -> Option<&V>
    where
        Flow: GetValue,
    {
        self.values.get(flow.get_value_idx())
    }
}
