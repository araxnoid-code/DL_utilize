use std::{
    fmt::Display,
    marker::PhantomData,
    ops::Add,
    sync::atomic::{AtomicUsize, Ordering::Relaxed},
};

use ndarray::{ArrayD, IxDyn, RawArrayView, RawArrayViewMut, s};

fn main() {
    let add_a = AddToolkit::<'_, f32, f32, f32>::new(&1., &2.);
    let add_b = AddToolkit::new(&add_a, &3.);
    let add_c = AddToolkit::new(&add_b, &4.);
    let add_d = AddToolkit::new(&add_a, &add_b);
    let add_e = AddToolkit::new(&add_c, &add_d);

    let mut vec = Vec::new();
    add_e.forward(&mut vec);
    //
    println!("{:?}", vec);
}

trait DlFlow<Out> {
    fn forward(&self, cache: &mut Vec<Out>) -> usize;
}

impl DlFlow<f32> for f32 {
    fn forward(&self, cache: &mut Vec<f32>) -> usize {
        println!("step: {}", self);
        cache.push(*self);
        cache.len() - 1
    }
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
    Out: Display,
{
    fn forward(&self, cache: &mut Vec<Out>) -> usize {
        let load = self.out_idx.load(Relaxed);
        if load != 0 {
            {
                let out = cache.get(load - 1).unwrap();
                println!("loaded step: {}", out)
            }

            return load - 1;
        }
        let lhs_idx = self.lhs.forward(cache);
        let rhs_idx = self.rhs.forward(cache);
        let lhs = cache.get(lhs_idx).expect("Error Getting Lhs");

        let rhs = cache.get(rhs_idx).expect("Error Getting Rhs");

        let out = lhs + rhs;
        println!("step: {}", out);
        cache.push(out);
        let idx = cache.len();
        self.out_idx.store(idx, Relaxed);

        idx - 1
    }
}
