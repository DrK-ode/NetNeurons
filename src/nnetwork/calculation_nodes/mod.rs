use std::{cell::RefCell, fmt::Debug, rc::Rc};

type FloatType = f64;
type TensorShape = (usize,usize,usize);
pub type TensorShared = Rc<RefCell<Tensor>>;
pub type OpNodeShared = Rc<OpNode>;

#[derive(Default)]
pub struct Tensor {
    _parent_op: Option<OpNodeShared>,
    _child_op: Option<OpNodeShared>,
    _shape: TensorShape,
    _value: Vec<FloatType>,
    _derivative: Vec<FloatType>,
}

pub trait Operator {
    fn operate(&self, inp: &[TensorShared], out: &TensorShared);
    fn back_propagate(&self, inp: &[TensorShared], out: &TensorShared);
    fn symbol(&self) -> &str;
}

pub struct OpNode {
    _op: Box<dyn Operator>,
    _inp: Vec<TensorShared>,
    _out: TensorShared,
}

struct NetworkCalculation {
    _op_order: Vec<OpNodeShared>,
}

mod op_node;
mod tensor;
mod network_calculation;
