use ndarray::{Array1, Array2, Array3, Array4, Array5, Array6, ArrayD, ArrayViewD};

use crate::{InitFlow, InitTool, InitViewFlow, ToolFlow};

impl<Array, Ref> ToolFlow for InitFlow<Array, Ref> {
    type ValueType = Array;
    type RefType = Ref;
    type ReturnType = Array;
    fn get_value<'a>(
        &'a self,
        values: &'a Vec<Self::ValueType>,
        _: &'a Vec<Self::RefType>,
    ) -> &'a Self::ReturnType {
        &values[self.value_idx]
    }

    fn get_value_idx(&self) -> usize {
        self.value_idx
    }

    fn get_gradient_idx(&self) -> Option<usize> {
        None
    }
}

impl<Array, Ref> ToolFlow for InitViewFlow<Array, Ref> {
    type ValueType = Array;
    type RefType = Ref;
    type ReturnType = Ref;
    fn get_value<'a>(
        &'a self,
        _: &'a Vec<Self::ValueType>,
        refs: &'a Vec<Self::RefType>,
    ) -> &'a Self::ReturnType {
        &refs[self.value_idx]
    }

    fn get_value_idx(&self) -> usize {
        self.value_idx
    }

    fn get_gradient_idx(&self) -> Option<usize> {
        None
    }
}

//
impl<F, R> InitTool<Array1<F>, R, InitFlow<Array1<F>, R>> for Array1<F> {
    fn push(self, values: &mut Vec<Array1<F>>, _: &mut Vec<R>) -> InitFlow<Array1<F>, R> {
        values.push(self);
        InitFlow {
            value_idx: values.len() - 1,
            _phantom: Default::default(),
        }
    }
}

impl<F, R> InitTool<Array2<F>, R, InitFlow<Array2<F>, R>> for Array2<F> {
    fn push(self, values: &mut Vec<Array2<F>>, _: &mut Vec<R>) -> InitFlow<Array2<F>, R> {
        values.push(self);
        InitFlow {
            value_idx: values.len() - 1,
            _phantom: Default::default(),
        }
    }
}

impl<F, R> InitTool<Array3<F>, R, InitFlow<Array3<F>, R>> for Array3<F> {
    fn push(self, values: &mut Vec<Array3<F>>, _: &mut Vec<R>) -> InitFlow<Array3<F>, R> {
        values.push(self);
        InitFlow {
            value_idx: values.len() - 1,
            _phantom: Default::default(),
        }
    }
}

impl<F, R> InitTool<Array4<F>, R, InitFlow<Array4<F>, R>> for Array4<F> {
    fn push(self, values: &mut Vec<Array4<F>>, _: &mut Vec<R>) -> InitFlow<Array4<F>, R> {
        values.push(self);
        InitFlow {
            value_idx: values.len() - 1,
            _phantom: Default::default(),
        }
    }
}

impl<F, R> InitTool<Array5<F>, R, InitFlow<Array5<F>, R>> for Array5<F> {
    fn push(self, values: &mut Vec<Array5<F>>, _: &mut Vec<R>) -> InitFlow<Array5<F>, R> {
        values.push(self);
        InitFlow {
            value_idx: values.len() - 1,
            _phantom: Default::default(),
        }
    }
}

impl<F, R> InitTool<Array6<F>, R, InitFlow<Array6<F>, R>> for Array6<F> {
    fn push(self, values: &mut Vec<Array6<F>>, _: &mut Vec<R>) -> InitFlow<Array6<F>, R> {
        values.push(self);
        InitFlow {
            value_idx: values.len() - 1,
            _phantom: Default::default(),
        }
    }
}

impl<F, R> InitTool<ArrayD<F>, R, InitFlow<ArrayD<F>, R>> for ArrayD<F> {
    fn push(self, values: &mut Vec<ArrayD<F>>, _: &mut Vec<R>) -> InitFlow<ArrayD<F>, R> {
        values.push(self);
        InitFlow {
            value_idx: values.len() - 1,
            _phantom: Default::default(),
        }
    }
}

//
impl<'a, F, V> InitTool<V, ArrayViewD<'a, F>, InitViewFlow<V, ArrayViewD<'a, F>>>
    for ArrayViewD<'a, F>
{
    fn push(
        self,
        _: &mut Vec<V>,
        refs: &mut Vec<ArrayViewD<'a, F>>,
    ) -> InitViewFlow<V, ArrayViewD<'a, F>> {
        refs.push(self);
        InitViewFlow {
            value_idx: refs.len() - 1,
            _phantom: Default::default(),
        }
    }
}
