use std::{marker::PhantomData, ops::Add};

use dl_toolkit_project::{AddTool, Toolkit};
use ndarray::ArrayD;
use num_traits::Zero;

fn main() {
    let mut toolkit: Toolkit<f32, ArrayD<f32>, ArrayD<f32>> = Toolkit::new();

    let init_1 = toolkit.init_value(ArrayD::<f32>::ones(vec![2, 2]));
    let init_2 = toolkit.init_value(ArrayD::<f32>::ones(vec![2, 2]));
    let init_3 = toolkit.init_value(ArrayD::<f32>::ones(vec![2, 2]));

    let add_flow_1 = toolkit.submit(AddTool::new(&init_1, &init_2));

    let add_flow_2 = toolkit.submit(AddTool::new(&init_3, &add_flow_1));

    let add_flow_3 = toolkit.submit(AddTool::new(&add_flow_1, &add_flow_2));

    println!("{}", toolkit.get_value(&add_flow_3).unwrap());
}
