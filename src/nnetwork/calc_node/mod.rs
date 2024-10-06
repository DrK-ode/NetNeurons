mod access;
mod ctors;
mod operators;
mod types;
mod differentiation;

use std::{cell::RefCell, rc::Rc};

pub use types::*;

#[derive(Default)]
pub struct CalcNode {
    _parent_nodes: Option<Vec<CalcNodeShared>>,
    _shape: NodeShape,
    _vals: Vec<FloatType>,
    _grad: Vec<FloatType>,
    // Function that calculates and updates the gradients for its parents.
    _back_propagation: Option<Box<dyn Fn(CalcNodeShared)>>,
}

#[derive(Clone)]
pub struct CalcNodeShared {
    _node: Rc<RefCell<CalcNode>>,
}
