use dl_toolkit_project::{AddTool, Label, Toolkit};
use ndarray::Array2;

fn main() {
    let mut toolkit = Toolkit::<Array2<f32>, Array2<f32>, Label<Array2<f32>>>::new();

    let init_1 = toolkit.init_value(Array2::<f32>::ones([2, 2]));
    let init_2 = toolkit.init_value(Array2::<f32>::ones([2, 2]));
    let init_3 = toolkit.init_value(Array2::<f32>::ones([2, 2]));
    let init_4 = toolkit.init_value(Array2::<f32>::ones([2, 2]));

    let add_flow_1 = toolkit.submit(AddTool::new(&init_1, &init_2));
    let add_flow_2 = toolkit.submit(AddTool::new(&init_3, &add_flow_1));
    let add_flow_3 = toolkit.submit(AddTool::new(&add_flow_2, &init_4));

    let add_flow_4 = toolkit.submit(AddTool::new(&add_flow_3, &add_flow_2));
    let add_flow_5 = toolkit.submit(AddTool::new(&add_flow_4, &add_flow_1));

    toolkit.backward_from(&add_flow_5);

    println!("{}", toolkit.get_value(&add_flow_5).unwrap());
    println!("{}", toolkit.get_gradient(&add_flow_1).unwrap());
    println!("{}", toolkit.get_gradient(&add_flow_2).unwrap());
}
