use crate::RefNumNeg;
use ndarray::{Dimension, IntoNdProducer, Zip};
use std::ops::{AddAssign, Sub};

pub fn sub_forward<Rhs, Lhs, Array>(rhs: &Rhs, lhs: &Lhs) -> Array
where
    for<'a> &'a Rhs: Sub<&'a Lhs, Output = Array>,
{
    rhs.sub(lhs)
}

pub fn sub_backward<F, Rhs, Lhs, Grad, Dim>(
    lhs_grad: Option<&mut Lhs>,
    rhs_grad: Option<&mut Rhs>,
    gradient: &Grad,
) where
    for<'a> Lhs: AddAssign<&'a Grad>,
    for<'b> &'b mut Rhs: IntoNdProducer<Dim = Dim, Item = &'b mut F>,
    for<'c> &'c Grad: IntoNdProducer<Dim = Dim, Item = &'c F>,
    for<'e> &'e F: RefNumNeg,
    for<'d> F: AddAssign<<&'d F as RefNumNeg>::Out>,
    Dim: Dimension,
{
    if let Some(lhs_grad) = lhs_grad {
        lhs_grad.add_assign(gradient);
    }

    if let Some(rhs_grad) = rhs_grad {
        Zip::from(rhs_grad)
            .and(gradient)
            .for_each(|lhs_grad, grad| {
                lhs_grad.add_assign(grad.neg());
            });
    }
}
