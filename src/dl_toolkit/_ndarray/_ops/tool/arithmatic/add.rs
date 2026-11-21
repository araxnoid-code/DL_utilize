use crate::{
    AddFlow, AddTool, Label, NDArrayGetShape, NDArrayMark, ToolComponent, ToolFlow, add_forward,
};
// use ndarray::ArrayD;
use std::ops::{Add, AddAssign};

impl<'t, Lhs, Rhs, Array, Ref> ToolComponent<Array, Ref, AddFlow<Array, Ref>>
    for AddTool<'t, Lhs, Rhs>
where
    Lhs: ToolFlow<ValueType = Array, RefType = Ref>,
    Rhs: ToolFlow<ValueType = Array, RefType = Ref>,
    for<'a> &'a Lhs::ReturnType: Add<&'a Rhs::ReturnType, Output = Array>,
    Array: NDArrayGetShape,
{
    type LabelType = Label<Array, Ref>;
    fn tool_forward(&self, values: &Vec<Array>, refs: &Vec<Ref>) -> Array {
        let lhs = self.lhs.get_value(values, refs);
        let rhs = self.rhs.get_value(values, refs);
        add_forward(lhs, rhs)
    }

    fn shape(out: &Array) -> Vec<usize> {
        out.get_shape().to_vec()
    }

    fn into_flow(self, value_idx: usize, grad_idx: Option<usize>) -> Self::LabelType {
        Label::Add(AddFlow::new(
            self.lhs.get_value_idx(),
            self.lhs.get_gradient_idx(),
            self.rhs.get_value_idx(),
            self.rhs.get_gradient_idx(),
            value_idx,
            grad_idx,
        ))
    }

    fn copy_label_to_flow(label: Self::LabelType) -> AddFlow<Array, Ref> {
        match label {
            Label::Add(add_flow) => add_flow,
        }
    }
}

impl<Array, Ref> AddFlow<Array, Ref>
where
    Array: NDArrayMark,
{
    pub fn backward<Grad>(&self, gradients: &mut Vec<Option<Grad>>)
    where
        for<'a> Grad: AddAssign<&'a Grad>,
    {
        if let Some(grad_idx) = self.grad_idx {
            let grad = gradients[grad_idx]
                .take()
                .expect(&format!("Error In AddFlow<Array>::backward where Array: NDArrayMark\nexpected a gradient, but got None at index {}", grad_idx));

            if let Some(lhs_grad_idx) = self.lhs_grad_idx {
                let lhs_grad = gradients[lhs_grad_idx].as_mut().expect(&format!("Error In AddFlow<Array>::backward where Array: NDArrayMark\nexpected a Lhs gradient, but got None at index {}", lhs_grad_idx));
                lhs_grad.add_assign(&grad);
            }

            if let Some(rhs_grad_idx) = self.rhs_grad_idx {
                let lhs_grad = gradients[rhs_grad_idx].as_mut().expect(&format!("Error In AddFlow<Array>::backward where Array: NDArrayMark\nexpected a Rhs gradient, but got None at index {}", rhs_grad_idx));
                lhs_grad.add_assign(&grad);
            }

            gradients[grad_idx].replace(grad);
        }
    }
}

impl<Array, Ref> ToolFlow for AddFlow<Array, Ref> {
    type ValueType = Array;
    type RefType = Ref;
    type ReturnType = Array;

    fn get_value<'a>(
        &'a self,
        values: &'a Vec<Self::ValueType>,
        _: &'a Vec<Self::RefType>,
    ) -> &'a Self::ReturnType {
        &values[self.value_idx]
    }

    fn get_value_idx(&self) -> usize {
        self.value_idx
    }

    fn get_gradient_idx(&self) -> Option<usize> {
        self.grad_idx
    }
}
