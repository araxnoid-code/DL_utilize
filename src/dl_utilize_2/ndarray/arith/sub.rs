use ::ndarray::{Dimension, IntoNdProducer};
use num_traits::Float;

use crate::*;
use std::ops::{AddAssign, Sub};

impl<'a, LI, RI, Own> SubTool<'a, LI, RI, Own> {
    pub fn new<Lhs, Rhs>(
        lhs: &'a Lhs,
        rhs: &'a Rhs,
        grads: &mut Vec<Option<Own>>,
    ) -> SubTool<'a, LI, RI, Own>
    where
        Lhs: ToolFLow<Own = LI>,
        Rhs: ToolFLow<Own = RI>,
        Own: Gradients<Out = Own> + NDArrayGetShape,
        for<'b> &'b LI: Sub<&'b RI, Output = Own>,
    {
        let lhs_item = lhs.get_value();
        let rhs_item = rhs.get_value();
        let value = lhs_item.sub(rhs_item);
        let grad = <Own as Gradients>::zeros(value.shape());
        grads.push(Some(grad));

        Self {
            lhs: lhs_item,
            lhs_grad_idx: lhs.get_grad_idx(),
            rhs: rhs_item,
            rhs_grad_idx: rhs.get_grad_idx(),
            value,
            grad_idx: Some(grads.len() - 1),
        }
    }
}

impl<Lhs, Rhs, Own> ToolFLow for SubTool<'_, Lhs, Rhs, Own>
where
    Lhs: ToolFLow,
    Rhs: ToolFLow,
{
    type Own = Own;

    fn get_value<'a>(&'a self) -> &'a Own {
        &self.value
    }
    fn get_grad_idx(&self) -> Option<usize> {
        self.grad_idx
    }
}

impl<LI, RI, Own, Dim, F> NDArrayBackward<Own> for SubTool<'_, LI, RI, Own>
where
    Own: NDArrayMark,
    F: Float + AddAssign<F>,
    for<'a> &'a Own: IntoNdProducer<Dim = Dim, Item = &'a F>,
    for<'a> &'a mut Own: IntoNdProducer<Dim = Dim, Item = &'a mut F>,
    for<'a> Own: AddAssign<&'a Own>,
    Dim: Dimension,
{
    fn backward(&self, grads: &mut Vec<Option<Own>>) {
        if let Some(grad_idx) = self.grad_idx {
            let mut lhs_grad = self.lhs_grad_idx.and_then(|idx| grads[idx].take());
            let mut rhs_grad = self.rhs_grad_idx.and_then(|idx| grads[idx].take());
            let grad = grads[grad_idx].as_ref().unwrap();

            sub_backward(rhs_grad.as_mut(), lhs_grad.as_mut(), grad);

            // lhs_grad.zip()
        }
    }
}
