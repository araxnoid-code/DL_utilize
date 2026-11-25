use std::ops::Mul;

use crate::{Label, MulFlow, MulTool, NDArrayGetShape, ToolComponent, ToolFlow, mul_forward};

impl<'a, Array, Ref, Lhs, Rhs> ToolComponent<Array, Ref, MulFlow> for MulTool<'a, Lhs, Rhs>
where
    Array: NDArrayGetShape,
    Lhs: ToolFlow<Array, Ref>,
    Rhs: ToolFlow<Array, Ref>,
    for<'b> &'b Lhs::ReturnType: Mul<&'b Rhs::ReturnType, Output = Array>,
{
    fn tool_forward(&self, values: &Vec<Array>, refs: &Vec<Ref>) -> Array {
        let lhs = self.lhs.get_value(values, refs);
        let rhs = self.rhs.get_value(values, refs);
        mul_forward(lhs, rhs)
    }

    type LabelType = Label;
    fn into_flow(self, value_idx: usize, grad_idx: Option<usize>) -> MulFlow {
        MulFlow {
            value_idx,
            grad_idx,
            lhs_idx: self.lhs.get_value_idx(),
            lhs_grad_idx: self.lhs.get_gradient_idx(),
            rhs_idx: self.rhs.get_value_idx(),
            rhs_grad_idx: self.rhs.get_gradient_idx(),
        }
    }

    fn shape(out: &Array) -> Vec<usize> {
        out.get_shape()
    }
}

impl MulFlow {
    pub fn backward<A, R, G>(&self, values: &Vec<A>, refs: &Vec<R>, gradients: &Vec<G>) {}
}
