use crate::*;
use std::ops::Sub;

impl<'a, LI, RI, Own> SubTool<'a, LI, RI, Own> {
    pub fn new<Lhs, Rhs>(
        lhs: &'a Lhs,
        rhs: &'a Rhs,
        grads: &mut Vec<Own>,
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
        grads.push(grad);

        Self {
            lhs: lhs_item,
            lhs_grad: lhs.get_grad_idx(),
            rhs: rhs_item,
            rhs_grad: rhs.get_grad_idx(),
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
