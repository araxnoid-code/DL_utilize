use ndarray::{Dimension, IntoNdProducer, Zip};
use std::ops::{AddAssign, Div, Mul};

use crate::RefNumNeg;

pub fn div_forward<Rhs, Lhs, Array>(rhs: &Rhs, lhs: &Lhs) -> Array
where
    for<'a> &'a Rhs: Div<&'a Lhs, Output = Array>,
{
    rhs.div(lhs)
}

pub fn div_backward<F, Rhs, Lhs, RhsGrad, LhsGrad, Grad, Dim>(
    lhs_value: &Lhs,
    lhs_grad: Option<&mut LhsGrad>,
    rhs_value: &Rhs,
    rhs_grad: Option<&mut RhsGrad>,
    gradient: &Grad,
) where
    // VALUE
    for<'a> &'a Rhs: IntoNdProducer<Dim = Dim, Item = &'a F>,
    for<'a> &'a Lhs: IntoNdProducer<Dim = Dim, Item = &'a F>,
    // ITEM
    for<'a> &'a F: Div<&'a F, Output = F> + Mul<&'a F, Output = F> + RefNumNeg<Out = F>,
    for<'a> F: AddAssign<F>,
    // GRADIENT
    for<'a> &'a mut LhsGrad: IntoNdProducer<Dim = Dim, Item = &'a mut F>,
    for<'a> &'a mut RhsGrad: IntoNdProducer<Dim = Dim, Item = &'a mut F>,
    for<'a> &'a Grad: IntoNdProducer<Dim = Dim, Item = &'a F>,
    //
    Dim: Dimension,
{
    if let Some(lhs_grad) = lhs_grad {
        Zip::from(lhs_grad)
            .and(rhs_value)
            .and(gradient)
            .for_each(|lhs_grad, rhs_val, grad| {
                lhs_grad.add_assign(grad.div(rhs_val));
            });
    }

    if let Some(rhs_grad) = rhs_grad {
        Zip::from(rhs_grad)
            .and(lhs_value)
            .and(rhs_value)
            .and(gradient)
            .for_each(|rhs_grad, lhs_val, rhs_val, grad| {
                rhs_grad.add_assign(&(&lhs_val.neg() * &(rhs_val * rhs_val)) * grad);
            });
    }
}
