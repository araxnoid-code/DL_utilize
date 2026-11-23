use std::{hint::black_box, ops::AddAssign, time};

use dl_utilize::{AddTool, Label, Toolkit, div_backward, mul_backward, mul_forward, sub_backward};
use ndarray::{ArrayD, ArrayViewD, Zip};

fn main() {
    let a = ArrayD::<f32>::ones(vec![2]);
    let mut a_grad = ArrayD::<f32>::ones(vec![2]);
    let b = ArrayD::<f32>::ones(vec![2]);
    let mut b_grad = ArrayD::<f32>::ones(vec![2]);
    let grad = ArrayD::<f32>::ones(vec![2]);

    div_backward(
        &a,
        Some(&mut a_grad.view_mut()),
        &b.view(),
        Some(&mut b_grad),
        &grad,
    );
}
