use dl_utilize::{AddTool, SubTool, Toolkit};
use ndarray::{ArrayD, ArrayViewD};

fn main() {
    let mut toolkit = Toolkit::<ArrayD<f32>, ArrayViewD<'_, f32>, ArrayD<f32>>::new();

    let array_1 = ArrayD::<f32>::ones(vec![2, 2]);
    let init_1 = toolkit.init_value::<_, _>(array_1.view());
    // let init_2 = toolkit.init_value(ArrayD::<f32>::ones(vec![2, 2]));
    //     let init_3 = toolkit.init_value(ArrayD::<f32>::ones(vec![2, 2]));
    //     let init_4 = toolkit.init_value(ArrayD::<f32>::ones(vec![2, 2]));
    //
    //     let a = AddTool::new(&init_1, &init_2);
    //     let add_flow_1 = toolkit.submit::<_, _>(a);
    //     let add_flow_2 = toolkit.submit(AddTool::new(&init_3, &add_flow_1));
    //     let add_flow_3 = toolkit.submit(SubTool::new(&add_flow_2, &init_4));
    //
    //     let add_flow_4 = toolkit.submit(AddTool::new(&add_flow_3, &add_flow_2));
    //     let add_flow_5 = toolkit.submit(SubTool::new(&add_flow_4, &add_flow_1));
    //
    //     toolkit.backward_from(&add_flow_5);
    //
    //     println!("{}", toolkit.get_value(&add_flow_5).unwrap());
    //     println!("{}", toolkit.get_gradient(&add_flow_1).unwrap());
    //     println!("{}", toolkit.get_gradient(&add_flow_2).unwrap());
}
