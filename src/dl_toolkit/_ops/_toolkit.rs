use std::marker::PhantomData;

use crate::{CopyLabel, Gradient, ToolComponent, ToolFlow};

pub struct Toolkit<V, Grad, Label> {
    pub(crate) values: Vec<V>,
    pub(crate) gradients: Vec<Option<Grad>>,
    pub(crate) flows: Vec<Label>,
}

impl<V, Grad, Label> Toolkit<V, Grad, Label> {
    pub fn get_value<Flow>(&self, flow: &Flow) -> Option<&V>
    where
        Flow: ToolFlow,
    {
        self.values.get(flow.get_value_idx())
    }

    pub fn submit<C, FlowType>(&mut self, component: C) -> FlowType
    where
        C: ToolComponent<V, FlowType, LabelType = Label>,
        Grad: Gradient,
        Label: CopyLabel,
    {
        let value = component.tool_forward(&self.values);
        let shape = C::shape(&value);
        self.values.push(value);
        self.gradients.push(Some(Grad::zeros(shape)));

        let flow = component.into_flow(self.values.len() - 1, Some(self.gradients.len() - 1));
        self.flows.push(flow.copy_label());
        C::copy_label_to_flow(flow)
    }
}
