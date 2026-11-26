use crate::Gradients;

pub trait NDArrayGetShape {
    fn shape(&self) -> Vec<usize>;
}

pub fn get_grad<Grad>(grads: &Vec<Option<Grad>>, idx: usize) -> &Grad {
    grads[idx].as_ref().unwrap()
}

pub fn fill_ones_to_grad<Grad>(grads: &mut Vec<Option<Grad>>, idx: usize)
where
    Grad: Gradients,
{
    grads[idx].as_mut().unwrap().fill_ones();
}
