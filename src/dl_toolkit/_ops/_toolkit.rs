use std::marker::PhantomData;

use num_traits::{Float, One};

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
