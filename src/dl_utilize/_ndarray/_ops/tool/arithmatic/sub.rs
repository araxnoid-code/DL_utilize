use std::ops::{AddAssign, Sub};

use ndarray::{Dimension, IntoNdProducer};

use crate::{
    Gradient, Label, NDArrayGetShape, RefNumNeg, SubFlow, SubTool, ToolComponent, ToolFlow,
    sub_backward, sub_forward,
};

impl<'t, Lhs, Rhs, Array, Ref> ToolComponent<Array, Ref, SubFlow> for SubTool<'t, Lhs, Rhs>
where
    Array: NDArrayGetShape,
    Lhs: ToolFlow<Array, Ref>,
    Rhs: ToolFlow<Array, Ref>,
    for<'a> &'a Lhs::ReturnType: Sub<&'a Rhs::ReturnType, Output = Array>,
{
    type LabelType = Label;

    fn tool_forward(&self, values: &Vec<Array>, refs: &Vec<Ref>) -> Array {
        let lhs = self.lhs.get_value(values, refs);
        let rhs = self.rhs.get_value(values, refs);

        sub_forward(lhs, rhs)
    }

    fn into_flow(self, value_idx: usize, grad_idx: Option<usize>) -> SubFlow {
        SubFlow {
            lhs_idx: self.lhs.get_value_idx(),
            lhs_grad_idx: self.lhs.get_gradient_idx(),
            rhs_idx: self.rhs.get_value_idx(),
            rhs_grad_idx: self.rhs.get_gradient_idx(),
            value_idx,
            grad_idx,
        }
    }

    fn shape(out: &Array) -> Vec<usize> {
        out.get_shape()
    }
}

impl SubFlow {
    pub fn backward<Grad, Dim>(&self, gradients: &mut Vec<Option<Grad>>)
    where
        for<'a> Grad: AddAssign<&'a Grad> + Gradient,
        for<'a> &'a mut Grad: IntoNdProducer<Dim = Dim, Item = &'a mut Grad::F>,
        for<'a> &'a Grad: IntoNdProducer<Dim = Dim, Item = &'a Grad::F>,
        for<'a> Grad: AddAssign<<&'a Grad::F as RefNumNeg>::Out>,
        for<'a> &'a Grad::F: RefNumNeg,
        for<'a> Grad::F: AddAssign<<&'a Grad::F as RefNumNeg>::Out>,
        Dim: Dimension,
    {
        if let Some(grad_idx) = self.grad_idx {
            let grad = gradients[grad_idx]
                .take()
                .expect(&format!("Error In AddFlow<Array>::backward where Array: NDArrayMark\nexpected a gradient, but got None at index {}", grad_idx));

            let mut lhs_grad = self
                .lhs_grad_idx
                .and_then(|idx| Some(gradients[idx].take().unwrap()));

            let mut rhs_grad = self
                .rhs_grad_idx
                .and_then(|idx| Some(gradients[idx].take().unwrap()));

            sub_backward(lhs_grad.as_mut(), rhs_grad.as_mut(), &grad);

            self.lhs_grad_idx.inspect(|idx| {
                gradients[*idx].replace(lhs_grad.unwrap());
            });

            self.rhs_grad_idx.inspect(|idx| {
                gradients[*idx].replace(rhs_grad.unwrap());
            });
        }
    }
}

impl<Array, Ref> ToolFlow<Array, Ref> for SubFlow {
    type ReturnType = Array;
    fn get_gradient_idx(&self) -> Option<usize> {
        self.grad_idx
    }

    fn get_value_idx(&self) -> usize {
        self.value_idx
    }

    fn get_value<'a>(&'a self, values: &'a Vec<Array>, _: &'a Vec<Ref>) -> &'a Self::ReturnType {
        &values[self.value_idx]
    }

    fn to_label(&self) -> Label {
        Label::Sub(*self)
    }
}
