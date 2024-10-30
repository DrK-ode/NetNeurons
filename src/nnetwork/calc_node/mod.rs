mod access;
mod ctors;
mod operators;
mod types;
mod back_propagation;

use std::{cell::RefCell, rc::Rc};

pub use types::*;

/// Wrapper class for [CalcNodeCore]. This is the struct intended to be used.
#[derive(Clone)]
pub struct CalcNode {
    _node: Rc<RefCell<CalcNodeCore>>,
}

/// The struct that actually holds the data.
/// Only a few member functions are implemented for this struct, use the wrapper struct [CalcNode] instead.
#[derive(Default)]
pub struct CalcNodeCore {
    _parent_nodes: Vec<CalcNode>,
    _shape: NodeShape,
    _vals: Vec<FloatType>,
    _grad: Vec<FloatType>,
    // Function that calculates and updates the gradients for its parents.
    _back_propagation: Option<Box<dyn Fn(CalcNode)>>,
}
