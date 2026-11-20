use std::{marker::PhantomData, ops::Add};

use ndarray::{ArrayD, ArrayViewD};
use num_traits::Zero;

fn main() {
    let mut toolkit: Toolkit<f32, ArrayD<f32>, ArrayD<f32>> = Toolkit::new();

    let init_1 = toolkit.init_value(ArrayD::<f32>::ones(vec![2, 2]));
    // let init_2 = toolkit.init_value(value)

    let add_flow_1 = toolkit.submit(AddTool::new(
        &init_1,
        &ArrayD::<f32>::ones(vec![2, 2]).view(),
    ));

    let add_flow_2 = toolkit.submit(AddTool::new(
        &ArrayD::<f32>::ones(vec![2, 2]).view(),
        &add_flow_1,
    ));

    let add_flow_3 = toolkit.submit(AddTool::new(&add_flow_1, &add_flow_2));

    println!("{}", toolkit.values[add_flow_3.value_idx]);
}

#[derive(Clone, Copy)]
enum Label<F> {
    Add(AddFlow<F>),
}

pub struct AddTool<'a, Lhs, Rhs> {
    lhs: &'a Lhs,
    rhs: &'a Rhs,
}

impl<'a, Lhs, Rhs> AddTool<'a, Lhs, Rhs> {
    pub fn new(lhs: &'a Lhs, rhs: &'a Rhs) -> AddTool<'a, Lhs, Rhs> {
        Self { lhs, rhs }
    }
}

struct InitFlow<F> {
    value_idx: usize,
    _phantom: PhantomData<F>,
}

impl<F> GetValue for InitFlow<F> {
    type ReturnType = ArrayD<F>;
    type ValueType = ArrayD<F>;
    fn get_value<'a>(&'a self, values: &'a Vec<Self::ValueType>) -> &'a Self::ReturnType {
        &values[self.value_idx]
    }
}

#[derive(Clone, Copy)]
pub struct AddFlow<F> {
    value_idx: usize,
    grad_idx: usize,
    _phantom: PhantomData<F>,
}

impl<F> AddFlow<F> {
    pub fn new(value_idx: usize, grad_idx: usize) -> AddFlow<F> {
        Self {
            value_idx,
            grad_idx,
            _phantom: Default::default(),
        }
    }
}

trait GetValue {
    type ReturnType;
    type ValueType;
    fn get_value<'a>(&'a self, values: &'a Vec<Self::ValueType>) -> &'a Self::ReturnType;
}

impl<F> GetValue for ArrayD<F> {
    type ReturnType = Self;
    type ValueType = ArrayD<F>;
    fn get_value<'a>(&'a self, _: &'a Vec<Self::ValueType>) -> &'a Self::ReturnType {
        self
    }
}

impl<'b, F> GetValue for ArrayViewD<'b, F> {
    type ReturnType = Self;
    type ValueType = ArrayD<F>;
    fn get_value<'a>(&'a self, _: &'a Vec<Self::ValueType>) -> &'a Self::ReturnType {
        self
    }
}

impl<F> GetValue for AddFlow<F> {
    type ValueType = ArrayD<F>;
    type ReturnType = ArrayD<F>;
    fn get_value<'a>(&'a self, values: &'a Vec<Self::ValueType>) -> &'a Self::ReturnType {
        &values[self.value_idx]
    }
}

impl<'t, Lhs, Rhs, F> ToolComponent<F, ArrayD<F>, AddFlow<F>> for AddTool<'t, Lhs, Rhs>
where
    Lhs: GetValue<ValueType = ArrayD<F>>,
    Rhs: GetValue<ValueType = ArrayD<F>>,
    for<'a> &'a Lhs::ReturnType: Add<&'a Rhs::ReturnType, Output = ArrayD<F>>,
{
    fn tool_forward(&self, values: &Vec<ArrayD<F>>) -> ArrayD<F> {
        let lhs = self.lhs.get_value(values);
        let rhs = self.rhs.get_value(values);
        lhs.add(rhs)
    }

    fn shape(out: &ArrayD<F>) -> Vec<usize> {
        out.shape().to_vec()
    }

    fn create_flow(value_idx: usize, grad_idx: usize) -> Label<F> {
        Label::Add(AddFlow::new(value_idx, grad_idx))
    }

    fn copy_label_to_flow(label: Label<F>) -> AddFlow<F> {
        match label {
            Label::Add(add_flow) => add_flow,
        }
    }
}

trait ToolComponent<F, Out, FlowType> {
    fn tool_forward(&self, values: &Vec<Out>) -> Out;
    fn shape(out: &Out) -> Vec<usize>;
    fn create_flow(value_idx: usize, grad_idx: usize) -> Label<F>;
    fn copy_label_to_flow(label: Label<F>) -> FlowType;
}

trait Gradient {
    fn zeros(shape: Vec<usize>) -> Self;
}

impl<F> Gradient for ArrayD<F>
where
    F: Clone + Zero,
{
    fn zeros(shape: Vec<usize>) -> Self {
        ArrayD::zeros(shape)
    }
}

struct Toolkit<F, V, Grad> {
    values: Vec<V>,
    gradients: Vec<Grad>,
    flows: Vec<Label<F>>,
}

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

        let flow = C::create_flow(self.values.len() - 1, self.gradients.len() - 1);
        self.flows.push(flow);
        C::copy_label_to_flow(flow)
    }
}
