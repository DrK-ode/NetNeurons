use std::rc::Rc;

use super::val_node::ValNodeShared;

pub type OpNodeShared = Rc<OpNode>;

enum NodeOp{
    Exp,
    Log,
    Neg,
    Add,
    Mul,
    Pow,
}

enum NodeData{
    Single(ValNodeShared),
    Many(Vec<ValNodeShared>),
}

struct OpNode{
    _op: NodeOp,
    _inp: NodeData,
    _out: NodeData,
}