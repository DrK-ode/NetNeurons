mod access;
mod ctors;
mod operators;
mod types;
mod back_propagation;

use std::{cell::RefCell, rc::Rc};

pub use types::*;

#[derive(Default)]
pub struct CalcNodeCore {
    _parent_nodes: Option<Vec<CalcNode>>,
    _shape: NodeShape,
    _vals: Vec<FloatType>,
    _grad: Vec<FloatType>,
    // Function that calculates and updates the gradients for its parents.
    _back_propagation: Option<Box<dyn Fn(CalcNode)>>,
}

#[derive(Clone)]
pub struct CalcNode {
    _node: Rc<RefCell<CalcNodeCore>>,
}
