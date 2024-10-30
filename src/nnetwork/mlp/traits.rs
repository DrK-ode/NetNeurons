use std::fmt::Display;

use crate::nnetwork::{CalcNode, NodeShape};

pub trait Layer: Parameters + Display {
    /// The shape determines what input shapes the layer accepts as well as the output shape it will give. For some [Layer]s it makes no sense to have a shape. If it makes sense though, this function must be overidden.
    fn shape(&self) -> Option<NodeShape> {
        None
    }

    /// Calculates the output given an input.
    fn forward(&self, inp: &CalcNode) -> CalcNode;

    /// All [Layer]s must have a name
    fn layer_name(&self) -> &str;
}

/// Object implementing this trait must supply iterators to all its parameters, in arbitrary, but fixed, order.
pub trait Parameters {
    fn param_iter(&self) -> Box<dyn Iterator<Item = &CalcNode> + '_>;
    fn param_iter_mut(&mut self) -> Box<dyn Iterator<Item = &mut CalcNode> + '_>;
}
