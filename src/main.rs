use std::{hint::black_box, ops::AddAssign, time};

use dl_utilize::{AddTool, Label, Toolkit, mul_backward, mul_forward, sub_backward};
use ndarray::{ArrayD, ArrayViewD, Zip};

fn main() {
    //     let mut toolkit = Toolkit::<ArrayD<f32>, ArrayD<f32>, ArrayViewD<'_, f32>, Label>::new();
    //
    //     let array_1 = ArrayD::<f32>::ones(vec![2, 2]);
    //
    //     let init_1 = toolkit.init_value(array_1.view());
    //     let init_2 = toolkit.init_value(ArrayD::<f32>::ones(vec![2, 2]));
    //     let init_3 = toolkit.init_value(ArrayD::<f32>::ones(vec![2, 2]));
    //     let init_4 = toolkit.init_value(ArrayD::<f32>::ones(vec![2, 2]));
    //
    //     let a = AddTool::new(&init_1, &init_2);
    //     let add_flow_1 = toolkit.submit(a);
    //     let add_flow_2 = toolkit.submit(AddTool::new(&init_3, &add_flow_1));
    //     let add_flow_3 = toolkit.submit(AddTool::new(&add_flow_2, &init_4));
    //
    //     let add_flow_4 = toolkit.submit(AddTool::new(&add_flow_3, &add_flow_2));
    //     let add_flow_5 = toolkit.submit(AddTool::new(&add_flow_4, &add_flow_1));
    //
    //     toolkit.backward_from(&add_flow_5);
    //
    //     println!("{}", toolkit.get_value(&add_flow_5).unwrap());
    //     println!("{}", toolkit.get_gradient(&add_flow_1).unwrap());
    //     println!("{}", toolkit.get_gradient(&add_flow_2).unwrap());

    let a = ArrayD::<f32>::ones(vec![20]);
    let mut b = ArrayD::<f32>::ones(vec![20]);

    mul_forward(&a, &b);
    mul_forward(&a.view(), &b);
    mul_forward(&a.view(), &b.view());
    mul_forward(&a.view(), &b.view_mut());

    let a = ArrayD::<f32>::ones(vec![20]);
    let mut a_grad = ArrayD::<f32>::ones(vec![20]);
    let b = ArrayD::<f32>::ones(vec![20]);
    let mut b_grad = ArrayD::<f32>::ones(vec![20]);
    let grad = ArrayD::<f32>::ones(vec![20]);

    mul_backward(&a, Some(&mut a_grad), &b, Some(&mut b_grad), &grad);

    mul_backward(
        &a.view(),
        Some(&mut a_grad),
        &b,
        Some(&mut b_grad.view_mut()),
        &grad.view(),
    );

    // mul_backward(Some(&mut a), Some(&mut b.view_mut()), &c.view());

    // f(&mut 10, 20);
}
