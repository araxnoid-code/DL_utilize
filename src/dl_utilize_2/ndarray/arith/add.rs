use crate::*;
use std::ops::{Add, AddAssign};

impl<'a, LI, RI, Own> AddTool<'a, LI, RI, Own> {
    pub fn new<Lhs, Rhs>(
        lhs: &'a Lhs,
        rhs: &'a Rhs,
        grads: &mut Vec<Option<Own>>,
    ) -> AddTool<'a, LI, RI, Own>
    where
        Lhs: ToolFLow<Own = LI>,
        Rhs: ToolFLow<Own = RI>,
        Own: Gradients<Out = Own> + NDArrayGetShape,
        for<'b> &'b LI: Add<&'b RI, Output = Own>,
    {
        let lhs_item = lhs.get_value();
        let rhs_item = rhs.get_value();
        let value = lhs_item.add(rhs_item);

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

impl<LI, RI, Own> ToolFLow for AddTool<'_, LI, RI, Own>
where
    LI: ToolFLow,
    RI: ToolFLow,
{
    type Own = Own;

    fn get_value<'a>(&'a self) -> &'a Own {
        &self.value
    }
    fn get_grad_idx(&self) -> Option<usize> {
        self.grad_idx
    }
}

impl<LI, RI, Own> NDArrayBackward<Own> for AddTool<'_, LI, RI, Own>
where
    for<'a> Own: AddAssign<&'a Own>,
{
    fn backward(&self, grads: &mut Vec<Option<Own>>) {
        if let Some(grad_idx) = self.grad_idx {
            let mut lhs_grad = self.lhs_grad_idx.and_then(|idx| grads[idx].take());
            let mut rhs_grad = self.rhs_grad_idx.and_then(|idx| grads[idx].take());
            let grad = grads[grad_idx].as_ref().unwrap();

            add_backward(lhs_grad.as_mut(), rhs_grad.as_mut(), grad);

            self.lhs_grad_idx
                .zip(lhs_grad)
                .and_then(|(idx, grad)| grads[idx].replace(grad));

            self.rhs_grad_idx
                .zip(rhs_grad)
                .and_then(|(idx, grad)| grads[idx].replace(grad));
        }
    }
}
