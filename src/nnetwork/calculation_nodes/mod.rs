use std::{cell::RefCell, fmt::Debug, ops::Deref, rc::Rc};

pub type FloatType = f64;
pub type TensorShape = (usize,usize,usize);
pub type OpNodeShared = Rc<OpNode>;

#[derive(Debug, PartialEq)]
pub enum VecOrientation {
    Column,
    Row,
}

#[derive(Debug, PartialEq)]
pub enum TensorType {
    None,
    Scalar,
    Vector(VecOrientation),
    Matrix,
    Tensor,
}

#[derive(Debug,Default,Clone)]
pub struct TensorShared{
    _tensor: Rc<RefCell<Tensor>>
}

#[derive(Default)]
pub struct Tensor {
    _parent_op: Option<OpNodeShared>,
    _child_op: Option<OpNodeShared>,
    _shape: TensorShape,
    _value: Vec<FloatType>,
    _derivative: Vec<FloatType>,
}

pub trait Operator {
    fn evaluate(&self, inp: &[TensorShared], out: &TensorShared);
    fn back_propagate(&self, inp: &[TensorShared], out: &TensorShared);
    fn symbol(&self) -> &str;
    fn output_shape(&self, input: &[TensorShared]) -> Option<TensorShape>;
}

pub struct OpNode {
    _op: Box<dyn Operator>,
    _inp: Vec<TensorShared>,
    _out: TensorShared,
}

pub struct NetworkCalculation {
    _op_order: Vec<OpNodeShared>,
}

mod op_node;
mod tensor;
mod network_calculation;
