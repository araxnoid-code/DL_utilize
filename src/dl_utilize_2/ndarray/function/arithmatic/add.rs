use std::ops::{Add, AddAssign};

pub fn add_forward<Rhs, Lhs, Array>(rhs: &Rhs, lhs: &Lhs) -> Array
where
    for<'a> &'a Rhs: Add<&'a Lhs, Output = Array>,
{
    rhs.add(lhs)
}

pub fn add_backward<Rhs, Lhs, Grad>(
    lhs_grad: Option<&mut Lhs>,
    rhs_grad: Option<&mut Rhs>,
    gradient: &Grad,
) where
    for<'a> Rhs: AddAssign<&'a Grad>,
    for<'b> Lhs: AddAssign<&'b Grad>,
{
    if let Some(rhs_grad) = rhs_grad {
        rhs_grad.add_assign(gradient);
    }

    if let Some(lhs_grad) = lhs_grad {
        lhs_grad.add_assign(gradient);
    }
}
