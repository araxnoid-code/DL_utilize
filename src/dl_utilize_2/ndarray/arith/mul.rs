use std::ops::{AddAssign, Mul};

use ndarray::{Dimension, IntoNdProducer};

use crate::{Backward, Gradients, MulTool, NDArrayGetShape, ToolFLow, mul_backward};

impl<'a, LI, RI, Own> MulTool<'a, LI, RI, Own> {
    pub fn new<Rhs, Lhs>(
        lhs: &'a Lhs,
        rhs: &'a Rhs,
        grads: &mut Vec<Option<Own>>,
    ) -> MulTool<'a, LI, RI, Own>
    where
        Lhs: ToolFLow<ValueType = LI>,
        Rhs: ToolFLow<ValueType = RI>,
        Own: Gradients<Out = Own> + NDArrayGetShape,
        for<'b> &'b LI: Mul<&'b RI, Output = Own>,
    {
        let lhs_item = lhs.get_value();
        let rhs_item = rhs.get_value();

        let forward = lhs_item.mul(rhs_item);
        grads.push(Some(<Own as Gradients>::zeros(forward.shape())));

        Self {
            lhs: lhs_item,
            lhs_grad_idx: lhs.get_grad_idx(),
            rhs: rhs_item,
            rhs_grad_idx: rhs.get_grad_idx(),
            value: forward,
            grad_idx: Some(grads.len() - 1),
        }
    }
}

impl<LI, RI, Own> ToolFLow for MulTool<'_, LI, RI, Own> {
    type ValueType = Own;
    fn get_grad_idx(&self) -> Option<usize> {
        self.grad_idx
    }

    fn get_value<'a>(&'a self) -> &'a Self::ValueType {
        &self.value
    }
}

impl<F, Lhs, LhsItem, Rhs, RhsItem, Own, Dim> Backward<Own> for MulTool<'_, Lhs, Rhs, Own>
where
    for<'a> &'a LhsItem: IntoNdProducer<Dim = Dim, Item = &'a F>,
    for<'a> &'a RhsItem: IntoNdProducer<Dim = Dim, Item = &'a F>,
    for<'a> &'a F: Mul<&'a F, Output = F>,
    F: AddAssign<F>,
    Lhs: ToolFLow<ValueType = LhsItem>,
    Rhs: ToolFLow<ValueType = RhsItem>,
    for<'a> &'a Own: IntoNdProducer<Dim = Dim, Item = &'a F>,
    for<'a> &'a mut Own: IntoNdProducer<Dim = Dim, Item = &'a mut F>,
    Dim: Dimension,
{
    fn backward(&self, grads: &mut Vec<Option<Own>>) {
        if let Some(grad_idx) = self.grad_idx {
            let lhs_value = self.lhs.get_value();
            let rhs_value = self.rhs.get_value();

            let mut lhs_grad = self.lhs_grad_idx.and_then(|idx| grads[idx].take());
            let mut rhs_grad = self.rhs_grad_idx.and_then(|idx| grads[idx].take());
            let grad = grads[grad_idx].as_ref().unwrap();

            mul_backward(
                lhs_value,
                lhs_grad.as_mut(),
                rhs_value,
                rhs_grad.as_mut(),
                grad,
            );

            self.lhs_grad_idx
                .zip(lhs_grad)
                .and_then(|(idx, grad)| grads[idx].replace(grad));

            self.rhs_grad_idx
                .zip(rhs_grad)
                .and_then(|(idx, grad)| grads[idx].replace(grad));
        }
    }
}
