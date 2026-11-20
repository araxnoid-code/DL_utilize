use crate::{Gradient, InitFlow, ToolComponent, Toolkit};

impl<F, V, Grad> Toolkit<F, V, Grad> {
    pub fn new() -> Toolkit<F, V, Grad> {
        Self {
            values: Vec::new(),
            gradients: Vec::new(),
            flows: Vec::new(),
        }
    }

    pub fn init_value(&mut self, value: V) -> InitFlow<F> {
        self.values.push(value);
        InitFlow {
            value_idx: self.values.len() - 1,
            _phantom: Default::default(),
        }
    }

    pub fn submit<C, FlowType>(&mut self, component: C) -> FlowType
    where
        F: Copy,
        C: ToolComponent<F, V, FlowType>,
        Grad: Gradient,
    {
        let value = component.tool_forward(&self.values);
        let shape = C::shape(&value);
        self.values.push(value);
        self.gradients.push(Grad::zeros(shape));

        let flow = component.into_flow(self.values.len() - 1, self.gradients.len() - 1);
        self.flows.push(flow);
        C::copy_label_to_flow(flow)
    }
}
