use num_traits::{Float, One};

use crate::{CopyLabel, Gradient, InitFlow, InitTool, ToolComponent, ToolFlow};

pub struct Toolkit<V, Grad, Ref, Label> {
    pub(crate) values: Vec<V>,
    pub(crate) gradients: Vec<Option<Grad>>,
    pub(crate) refs: Vec<Ref>,
    pub(crate) flows: Vec<Label>,
}

impl<V, Grad, Ref, Label> Toolkit<V, Grad, Ref, Label> {
    pub fn init_value<InitValue, Flow>(&mut self, value: InitValue) -> Flow
    where
        InitValue: InitTool<V, Ref, Flow>,
    {
        value.push(&mut self.values, &mut self.refs)
    }

    pub fn get_value<Flow>(&self, flow: &Flow) -> Option<&V>
    where
        Flow: ToolFlow,
    {
        self.values.get(flow.get_value_idx())
    }

    pub fn grad_into_ones<Flow>(&mut self, flow: &Flow)
    where
        Flow: ToolFlow,
        Grad: Gradient,
        Grad::F: Float,
    {
        if let Some(grad_idx) = flow.get_gradient_idx() {
            self.gradients[grad_idx]
                .as_mut()
                .expect("Error In Toolkit::grad_into_ones. Gradient is None")
                .fill(One::one());
        }
    }

    pub fn get_gradient<Flow>(&self, flow: &Flow) -> Option<&Grad>
    where
        Flow: ToolFlow,
    {
        if let Some(grad_idx) = flow.get_gradient_idx() {
            self.gradients[grad_idx].as_ref()
        } else {
            None
        }
    }

    pub fn submit<C, FlowType>(&mut self, component: C) -> FlowType
    where
        C: ToolComponent<V, Ref, FlowType, LabelType = Label>,
        Grad: Gradient,
        Label: CopyLabel,
    {
        let value = component.tool_forward(&self.values, &self.refs);
        let shape = C::shape(&value);
        self.values.push(value);
        self.gradients.push(Some(Grad::zeros(shape)));

        let flow = component.into_flow(self.values.len() - 1, Some(self.gradients.len() - 1));
        self.flows.push(flow.copy_label());
        C::copy_label_to_flow(flow)
    }
}
