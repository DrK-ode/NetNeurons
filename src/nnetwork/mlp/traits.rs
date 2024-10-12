use std::fmt::Display;

use crate::nnetwork::{CalcNode, NodeShape};

pub trait Layer: Parameters + Display {
    fn shape(&self) -> Option<NodeShape> {
        None
    }

    fn forward(&self, inp: &CalcNode) -> CalcNode;

    fn layer_name(&self) -> &str;
}

pub trait Parameters {
    fn param_iter(&self) -> Box<dyn Iterator<Item = &CalcNode> + '_>;
    fn param_iter_mut(&mut self) -> Box<dyn Iterator<Item = &mut CalcNode> + '_>;
}
