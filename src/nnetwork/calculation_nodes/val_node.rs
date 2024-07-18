use std::{cell::RefCell, fmt::Display, ops::Add, rc::Rc};

use super::op_node::OpNodeShared;

type FloatType = f64;
pub type ValNodeShared = Rc<RefCell<ValNode>>;

enum NodeValue {
    Scalar(FloatType),
    Vector(Vec<FloatType>),
}

pub struct ValNode {
    _from_op: Option<OpNodeShared>,
    _to_op: Option<OpNodeShared>,
    _value: NodeValue,
    _derivative: Option<NodeValue>,
}

impl ValNode {
    pub fn new(value: NodeValue) -> ValNode {
        ValNode {
            _from_op: None,
            _to_op: None,
            _value: value,
            _derivative: None,
        }
    }
}

impl Display for ValNode{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

impl Add for ValNode{
    type Output = ValNode;

    fn add(self, rhs: Self) -> Self::Output {
        todo!()
    }
}