use dl_utilize::{AddTool, Backward, SubTool, ToolFLow, fill_ones_to_grad, get_grad};
use ndarray::ArrayD;

fn main() {
    let array_1 = ArrayD::<f32>::from_shape_vec(vec![2], vec![0., 0.]).unwrap();
    let array_2 = ArrayD::<f32>::from_shape_vec(vec![2], vec![1., 1.]).unwrap();
    let view_2 = array_2.view();
    let array_3 = ArrayD::<f32>::from_shape_vec(vec![2], vec![2., 2.]).unwrap();
    let array_4 = ArrayD::<f32>::from_shape_vec(vec![2], vec![3., 3.]).unwrap();

    let mut grads = Vec::new();
    let f1 = AddTool::new(&array_1, &view_2, &mut grads);
    println!("f1: {}", f1.get_value());
    let f2 = AddTool::new(&array_3, &f1, &mut grads);
    println!("f2: {}", f2.get_value());
    let f3 = SubTool::new(&f2, &array_4, &mut grads);
    println!("f3: {}", f3.get_value());
    let f4 = AddTool::new(&f3, &f2, &mut grads);
    println!("f4: {}", f4.get_value());
    let f5 = SubTool::new(&f4, &f1, &mut grads);
    println!("f5: {}", f5.get_value());

    fill_ones_to_grad(&mut grads, f5.get_grad_idx().unwrap());

    f5.backward(&mut grads);
    f4.backward(&mut grads);
    f3.backward(&mut grads);
    f2.backward(&mut grads);
    f1.backward(&mut grads);

    println!("G f5: {}", get_grad(&grads, f5.get_grad_idx().unwrap()));
    println!("G f4: {}", get_grad(&grads, f4.get_grad_idx().unwrap()));
    println!("G f3: {}", get_grad(&grads, f3.get_grad_idx().unwrap()));
    println!("G f2: {}", get_grad(&grads, f2.get_grad_idx().unwrap()));
    println!("G f1: {}", get_grad(&grads, f1.get_grad_idx().unwrap()));
}
