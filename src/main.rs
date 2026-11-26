use dl_utilize::{AddTool, NDArrayBackward, SubTool, ToolFLow};
use ndarray::ArrayD;

fn main() {
    let array_1 = ArrayD::<f32>::from_shape_vec(vec![2], vec![0., 0.]).unwrap();
    let array_2 = ArrayD::<f32>::from_shape_vec(vec![2], vec![1., 1.]).unwrap();
    let view_2 = array_2.view();
    let array_3 = ArrayD::<f32>::from_shape_vec(vec![2], vec![2., 2.]).unwrap();
    let array_4 = ArrayD::<f32>::from_shape_vec(vec![2], vec![3., 3.]).unwrap();

    let mut grads = Vec::new();

    let add = AddTool::new(&array_1, &view_2, &mut grads);

    let sub = SubTool::new(&add, &array_3, &mut grads);
    sub.backward(&mut grads);

    //
    //     let add = AddTool::new(&sub, &array_4, &mut grad);
}
