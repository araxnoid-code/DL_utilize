use std::{
    fmt::Display,
    marker::PhantomData,
    ops::Add,
    sync::atomic::{AtomicUsize, Ordering::Relaxed},
};

use ndarray::{ArrayD, IxDyn, RawArrayView, RawArrayViewMut, s};
use num_traits::Float;

fn main() {
    let add_a = AddToolkit::<'_, f32, f32, f32>::new(&1., &2.);
    let add_b = AddToolkit::new(&add_a, &3.);
    let add_c = AddToolkit::new(&add_b, &4.);
    let add_d = AddToolkit::new(&add_a, &add_b);
    let add_e = AddToolkit::new(&add_c, &add_d);

    let mut cache = Vec::new();
    let mut grad = Vec::new();
    add_e.forward_flow(&mut cache, &mut grad);
    //
    println!("{:?}", cache);

    *grad.get_mut(add_e.out_idx.load(Relaxed) - 1).unwrap() = 1.;
    println!("{:?}", grad);
}

trait DlFlow<Out> {
    fn forward_flow(&self, cache: &mut Vec<Out>, grad: &mut Vec<Out>) -> usize;
}

impl DlFlow<f32> for f32 {
    fn forward_flow(&self, cache: &mut Vec<f32>, grad: &mut Vec<f32>) -> usize {
        println!("step: {}", self);
        cache.push(*self);
        grad.push(0.);
        cache.len() - 1
    }

    // fn backward_flow(&self, cache: &mut Vec<f32>, grad: &mut Vec<f32>) {}
}

struct AddToolkit<'a, Lhs, Rhs, Out>
where
    Lhs: DlFlow<Out>,
    Rhs: DlFlow<Out>,
{
    lhs: &'a Lhs,
    rhs: &'a Rhs,
    out_idx: AtomicUsize,
    _phantom: PhantomData<Out>,
}

impl<'a, Lhs, Rhs, Out> AddToolkit<'a, Lhs, Rhs, Out>
where
    Lhs: DlFlow<Out>,
    Rhs: DlFlow<Out>,
{
    pub fn new(lhs: &'a Lhs, rhs: &'a Rhs) -> AddToolkit<'a, Lhs, Rhs, Out> {
        Self {
            lhs,
            rhs,
            out_idx: AtomicUsize::new(0),
            _phantom: PhantomData::default(),
        }
    }
}

impl<'a, Lhs, Rhs, Out> DlFlow<Out> for AddToolkit<'a, Lhs, Rhs, Out>
where
    Lhs: DlFlow<Out>,
    Rhs: DlFlow<Out>,
    for<'b> &'b Out: Add<&'b Out, Output = Out>,
    Out: Display + Float,
{
    fn forward_flow(&self, cache: &mut Vec<Out>, grad: &mut Vec<Out>) -> usize {
        let load = self.out_idx.load(Relaxed);
        if load != 0 {
            {
                let out = cache.get(load - 1).unwrap();
                println!("loaded step: {}", out)
            }

            return load - 1;
        }

        let lhs_idx = self.lhs.forward_flow(cache, grad);
        let rhs_idx = self.rhs.forward_flow(cache, grad);
        let lhs = cache.get(lhs_idx).expect("Error Getting Lhs");

        let rhs = cache.get(rhs_idx).expect("Error Getting Rhs");

        let out = lhs + rhs;
        println!("step: {}", out);
        cache.push(out);
        grad.push(Out::zero());
        let idx = cache.len();
        self.out_idx.store(idx, Relaxed);

        idx - 1
    }
}
