use ndarray::{Dimension, IntoNdProducer, Zip};
use std::ops::{AddAssign, Mul};

pub fn mul_forward<Rhs, Lhs, Array>(rhs: &Rhs, lhs: &Lhs) -> Array
where
    for<'a> &'a Rhs: Mul<&'a Lhs, Output = Array>,
{
    rhs.mul(lhs)
}

pub fn mul_backward<F, Rhs, Lhs, RhsGrad, LhsGrad, Grad, Dim>(
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
    for<'a> &'a F: Mul<&'a F, Output = F>,
    for<'a> &'a F: Mul<&'a F, Output = F>,
    for<'a> F: AddAssign<F>,
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
                let mul = rhs_val.mul(grad);
                lhs_grad.add_assign(mul);
            });
    }

    if let Some(rhs_grad) = rhs_grad {
        Zip::from(rhs_grad)
            .and(lhs_value)
            .and(gradient)
            .for_each(|rhs_grad, lhs_val, grad| {
                let mul = lhs_val.mul(grad);
                rhs_grad.add_assign(mul);
            });
    }
}
