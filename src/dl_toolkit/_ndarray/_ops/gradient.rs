use crate::Gradient;
use ndarray::{Array1, Array2, Array3, Array4, Array5, Array6, ArrayD};
use num_traits::Zero;

impl<F> Gradient for ArrayD<F>
where
    F: Clone + Zero,
{
    fn zeros(shape: Vec<usize>) -> Self {
        ArrayD::zeros(shape)
    }
}

impl<F> Gradient for Array1<F>
where
    F: Clone + Zero,
{
    fn zeros(shape: Vec<usize>) -> Self {
        let shape_arr: [usize; 1] = shape.try_into().unwrap();
        Array1::zeros(shape_arr)
    }
}

impl<F> Gradient for Array2<F>
where
    F: Clone + Zero,
{
    fn zeros(shape: Vec<usize>) -> Self {
        let shape_arr: [usize; 2] = shape.try_into().unwrap();
        Array2::zeros(shape_arr)
    }
}

impl<F> Gradient for Array3<F>
where
    F: Clone + Zero,
{
    fn zeros(shape: Vec<usize>) -> Self {
        let shape_arr: [usize; 3] = shape.try_into().unwrap();
        Array3::zeros(shape_arr)
    }
}

impl<F> Gradient for Array4<F>
where
    F: Clone + Zero,
{
    fn zeros(shape: Vec<usize>) -> Self {
        let shape_arr: [usize; 4] = shape.try_into().unwrap();
        Array4::zeros(shape_arr)
    }
}

impl<F> Gradient for Array5<F>
where
    F: Clone + Zero,
{
    fn zeros(shape: Vec<usize>) -> Self {
        let shape_arr: [usize; 5] = shape.try_into().unwrap();
        Array5::zeros(shape_arr)
    }
}

impl<F> Gradient for Array6<F>
where
    F: Clone + Zero,
{
    fn zeros(shape: Vec<usize>) -> Self {
        let shape_arr: [usize; 6] = shape.try_into().unwrap();
        Array6::zeros(shape_arr)
    }
}
