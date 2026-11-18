use std::{fmt::Display, marker::PhantomData, ops::Add};

fn main() {
    let _toolkit = Toolkit {
        _phantom: Default::default(),
        f: AddOps::new(1, 1),
        data: 4,
        prev: Toolkit {
            _phantom: Default::default(),
            f: AddOps::new(2, 2),
            data: 3,
            prev: Toolkit {
                _phantom: Default::default(),
                f: AddOps::new(3, 3),
                data: 2,
                prev: Toolkit {
                    _phantom: Default::default(),
                    f: AddOps::new(4, 4),
                    data: 1,
                    prev: Init,
                },
            },
        },
    };

    _toolkit.prev_forward();
}

//

trait ToolkitTrait {
    fn prev_forward(&self) -> i32;
}

struct Init;
impl ToolkitTrait for Init {
    fn prev_forward(&self) -> i32 {
        println!("nothing");
        0
    }
}

struct Toolkit<Tool, Ops, F>
where
    Ops: OpsToolkit<F>,
{
    prev: Tool,
    f: Ops,
    data: i32,
    _phantom: PhantomData<F>,
}

impl<Tool, Ops, F> ToolkitTrait for Toolkit<Tool, Ops, F>
where
    Tool: ToolkitTrait,
    Ops: OpsToolkit<F>,
    F: Display,
{
    fn prev_forward(&self) -> i32 {
        let out = self.prev.prev_forward();
        println!("toolkit idx: {}", out);
        println!("output: {}", self.f.ops_forward());
        self.data
    }
}

// OPS
trait OpsToolkit<F> {
    fn ops_forward(&self) -> F;
}

struct AddOps<F> {
    lhs: F,
    rhs: F,
}

impl<F> AddOps<F> {
    pub fn new(lhs: F, rhs: F) -> AddOps<F> {
        Self { lhs, rhs }
    }
}

impl<F> OpsToolkit<F> for AddOps<F>
where
    F: Copy + Add<Output = F>,
{
    fn ops_forward(&self) -> F {
        self.lhs + self.rhs
    }
}
