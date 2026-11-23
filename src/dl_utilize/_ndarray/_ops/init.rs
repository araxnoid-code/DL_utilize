use ndarray::{Array1, Array2, Array3, Array4, Array5, Array6, ArrayD, ArrayViewD};

use crate::{InitFlow, InitTool, InitViewFlow, ToolFlow};

impl<Array, Ref> ToolFlow<Array, Ref> for InitFlow {
    type ReturnType = Array;
    fn get_value<'a>(&'a self, values: &'a Vec<Array>, _: &'a Vec<Ref>) -> &'a Self::ReturnType {
        &values[self.value_idx]
    }

    fn get_value_idx(&self) -> usize {
        self.value_idx
    }

    fn get_gradient_idx(&self) -> Option<usize> {
        None
    }

    fn to_label(&self) -> crate::Label {
        panic!()
    }
}

impl<Array, Ref> ToolFlow<Array, Ref> for InitViewFlow {
    type ReturnType = Ref;
    fn get_value<'a>(&'a self, _: &'a Vec<Array>, refs: &'a Vec<Ref>) -> &'a Self::ReturnType {
        &refs[self.value_idx]
    }

    fn get_value_idx(&self) -> usize {
        self.value_idx
    }

    fn get_gradient_idx(&self) -> Option<usize> {
        None
    }

    fn to_label(&self) -> crate::Label {
        panic!()
    }
}

//
impl<F, R> InitTool<Array1<F>, R, InitFlow> for Array1<F> {
    fn push(self, values: &mut Vec<Array1<F>>, _: &mut Vec<R>) -> InitFlow {
        values.push(self);
        InitFlow {
            value_idx: values.len() - 1,
        }
    }
}

impl<F, R> InitTool<Array2<F>, R, InitFlow> for Array2<F> {
    fn push(self, values: &mut Vec<Array2<F>>, _: &mut Vec<R>) -> InitFlow {
        values.push(self);
        InitFlow {
            value_idx: values.len() - 1,
        }
    }
}

impl<F, R> InitTool<Array3<F>, R, InitFlow> for Array3<F> {
    fn push(self, values: &mut Vec<Array3<F>>, _: &mut Vec<R>) -> InitFlow {
        values.push(self);
        InitFlow {
            value_idx: values.len() - 1,
        }
    }
}

impl<F, R> InitTool<Array4<F>, R, InitFlow> for Array4<F> {
    fn push(self, values: &mut Vec<Array4<F>>, _: &mut Vec<R>) -> InitFlow {
        values.push(self);
        InitFlow {
            value_idx: values.len() - 1,
        }
    }
}

impl<F, R> InitTool<Array5<F>, R, InitFlow> for Array5<F> {
    fn push(self, values: &mut Vec<Array5<F>>, _: &mut Vec<R>) -> InitFlow {
        values.push(self);
        InitFlow {
            value_idx: values.len() - 1,
        }
    }
}

impl<F, R> InitTool<Array6<F>, R, InitFlow> for Array6<F> {
    fn push(self, values: &mut Vec<Array6<F>>, _: &mut Vec<R>) -> InitFlow {
        values.push(self);
        InitFlow {
            value_idx: values.len() - 1,
        }
    }
}

impl<F, R> InitTool<ArrayD<F>, R, InitFlow> for ArrayD<F> {
    fn push(self, values: &mut Vec<ArrayD<F>>, _: &mut Vec<R>) -> InitFlow {
        values.push(self);
        InitFlow {
            value_idx: values.len() - 1,
        }
    }
}

//
impl<'a, F, V> InitTool<V, ArrayViewD<'a, F>, InitViewFlow> for ArrayViewD<'a, F> {
    fn push(self, _: &mut Vec<V>, refs: &mut Vec<ArrayViewD<'a, F>>) -> InitViewFlow {
        refs.push(self);
        InitViewFlow {
            value_idx: refs.len() - 1,
            // _phantom: Default::default(),
        }
    }
}
