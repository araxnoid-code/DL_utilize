use crate::RefNumNeg;
use ndarray::{Dimension, IntoNdProducer, Zip};
use std::ops::{AddAssign, Sub};

pub fn sub_forward<Rhs, Lhs, Array>(rhs: &Rhs, lhs: &Lhs) -> Array
where
    for<'a> &'a Rhs: Sub<&'a Lhs, Output = Array>,
{
    rhs.sub(lhs)
}

pub fn sub_backward<Rhs, Lhs, Grad, LhsItem, RhsItem, Dim>(
    rhs_grad: Option<&mut Rhs>,
    lhs_grad: Option<&mut Lhs>,
    gradient: &Grad,
) where
    for<'a> Rhs: AddAssign<&'a Grad>,
    for<'b> &'b mut Lhs: IntoNdProducer<Dim = Dim, Item = &'b mut LhsItem>,
    for<'c> &'c Grad: IntoNdProducer<Dim = Dim, Item = &'c RhsItem>,
    for<'e> &'e RhsItem: RefNumNeg,
    for<'d> LhsItem: AddAssign<<&'d RhsItem as RefNumNeg>::Out>,
    Dim: Dimension,
{
    if let Some(rhs_grad) = rhs_grad {
        rhs_grad.add_assign(gradient);
    }

    if let Some(lhs_grad) = lhs_grad {
        Zip::from(lhs_grad)
            .and(gradient)
            .for_each(|lhs_grad, grad| {
                lhs_grad.add_assign(grad.neg());
            });
    }
}
