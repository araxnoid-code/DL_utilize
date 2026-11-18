use dl_toolkit_project::{add_backward, add_forward};
use ndarray::ArrayD;

fn main() {
    let array_a = ArrayD::from_shape_vec(vec![2, 3], vec![1., 2., 3., 4., 5., 6.])
        .expect("Error Creating Array");
    let mut gradient_a = ArrayD::<f32>::zeros(vec![2, 3]);

    let array_b = ArrayD::from_shape_vec(vec![2, 3], vec![7., 8., 9., 10., 11., 12.])
        .expect("Error Creating Array");
    let mut gradient_b = ArrayD::<f32>::zeros(vec![2, 3]);

    let array_c = ArrayD::from_shape_vec(vec![2, 3], vec![7., 8., 9., 10., 11., 12.])
        .expect("Error Creating Array");
    let mut gradient_c = ArrayD::<f32>::zeros(vec![2, 3]);

    //

    let array_d = add_forward(&array_a, &array_b);
    let mut gradient_d = ArrayD::<f32>::zeros(vec![2, 3]);

    let _array_e = add_forward(&array_c, &array_d);
    let gradient_e = ArrayD::<f32>::ones(vec![2, 3]);

    //
    add_backward(Some(&mut gradient_c), Some(&mut gradient_d), &gradient_e);
    add_backward(Some(&mut gradient_a), Some(&mut gradient_b), &gradient_d);

    println!("{}", gradient_b);
    println!("{}", gradient_a);
}
