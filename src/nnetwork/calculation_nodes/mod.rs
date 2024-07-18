use std::{cell::RefCell, rc::Rc};

type FloatType = f64;
pub type ValNodeShared = Rc<RefCell<ValNode>>;
pub type OpNodeShared = Rc<OpNode>;

#[derive(Clone,Debug)]
enum NodeValue {
    Scalar(FloatType),
    Vector(Vec<FloatType>),
}

pub struct ValNode {
    _parent_op: Option<OpNodeShared>,
    _child_op: Option<OpNodeShared>,
    _value: Option<NodeValue>,
    _derivative: Option<NodeValue>,
}

impl ValNode {
    pub fn new() -> ValNode {
        ValNode {
            _parent_op: None,
            _child_op: None,
            _value: None,
            _derivative: None,
        }
    }
}

pub enum NodeOp {
    Exp,
    Log,
    Neg,
    Add,
    Mul,
    Pow,
}

pub enum NodeData {
    Single(ValNodeShared),
    Many(Vec<ValNodeShared>),
}

pub struct OpNode {
    _op: NodeOp,
    _inp: NodeData,
    _out: NodeData,
}

mod val_node;
mod op_node;
mod network_calculation;