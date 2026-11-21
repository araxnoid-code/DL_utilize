use dl_toolkit_project::{AddTool, Label, Toolkit};
use ndarray::{Array2, ArrayD};

fn main() {
    let mut toolkit = Toolkit::<Array2<f32>, Array2<f32>, Label<Array2<f32>>>::new();

    let init_1 = toolkit.init_value(Array2::<f32>::ones([2, 2]));
    let init_2 = toolkit.init_value(Array2::<f32>::ones([2, 2]));
    let init_3 = toolkit.init_value(Array2::<f32>::ones([2, 2]));

    let add_flow_1 = toolkit.submit(AddTool::new(&init_1, &init_2));
    let add_flow_2 = toolkit.submit(AddTool::new(&init_3, &add_flow_1));
    let add_flow_3 = toolkit.submit(AddTool::new(&add_flow_1, &add_flow_2));

    toolkit.backward();

    // println!("{}", toolkit.get_value(&add_flow_3).unwrap());
}
