use crate::{Gradient, InitTool, Label, ToolComponent, ToolFlow};
use num_traits::{Float, One};

pub struct Toolkit<V, Grad, Ref> {
    pub(crate) values: Vec<V>,
    pub(crate) gradients: Vec<Option<Grad>>,
    pub(crate) refs: Vec<Ref>,
    pub(crate) label_flows: Vec<Label>,
}

impl<Arr, Grad, Ref> Toolkit<Arr, Grad, Ref> {
    pub fn init_value<InitValue, Flow>(&mut self, value: InitValue) -> Flow
    where
        InitValue: InitTool<Arr, Ref, Flow>,
    {
        value.push(&mut self.values, &mut self.refs)
    }

    pub fn get_value<Flow>(&self, flow: &Flow) -> Option<&Arr>
    where
        Flow: ToolFlow<Arr, Ref>,
    {
        self.values.get(flow.get_value_idx())
    }

    pub fn grad_into_ones<Flow>(&mut self, flow: &Flow)
    where
        Flow: ToolFlow<Arr, Ref>,
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
        Flow: ToolFlow<Arr, Ref>,
    {
        if let Some(grad_idx) = flow.get_gradient_idx() {
            self.gradients[grad_idx].as_ref()
        } else {
            None
        }
    }

    pub fn submit<C, FlowType>(&mut self, component: C) -> FlowType
    where
        C: ToolComponent<Arr, Ref, FlowType, LabelType = Label>,
        Grad: Gradient,
        FlowType: ToolFlow<Arr, Ref>,
    {
        let value = component.tool_forward(&self.values, &self.refs);
        let shape = C::shape(&value);
        self.values.push(value);
        self.gradients.push(Some(Grad::zeros(shape)));

        let flow = component.into_flow(self.values.len() - 1, Some(self.gradients.len() - 1));
        self.label_flows.push(flow.to_label());
        flow
    }
}
