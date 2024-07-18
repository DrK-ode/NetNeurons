use std::{cell::RefCell, rc::Rc};

use super::*;

impl OpNode {
    pub fn unary( op: NodeOp, inp: &ValNodeShared) -> ValNodeShared {
        let out = Rc::new(RefCell::new(ValNode::new()));
        let op = Rc::new(OpNode {
            _op: op,
            _inp: NodeData::Single(inp.clone()),
            _out: NodeData::Single(out.clone()),
        });
        inp.borrow_mut()._child_op = Some(op.clone());
        out.borrow_mut()._parent_op = Some(op.clone());

        out
    }

    pub fn binary( op: NodeOp, inp1: &ValNodeShared, inp2: &ValNodeShared) -> ValNodeShared {
        let out = Rc::new(RefCell::new(ValNode::new()));
        let op = Rc::new(OpNode {
            _op: op,
            _inp: NodeData::Many(vec![inp1.clone(), inp2.clone()]),
            _out: NodeData::Single(out.clone()),
        });
        inp1.borrow_mut()._child_op = Some(op.clone());
        inp2.borrow_mut()._child_op = Some(op.clone());
        out.borrow_mut()._parent_op = Some(op.clone());

        out
    }

    pub fn multi( op: NodeOp, inp: &Vec<ValNodeShared>) -> ValNodeShared {
        let out = Rc::new(RefCell::new(ValNode::new()));
        let op = Rc::new(OpNode {
            _op: op,
            _inp: NodeData::Many(inp.clone()),
            _out: NodeData::Single(out.clone()),
        });
        inp.iter().for_each(|node| node.borrow_mut()._child_op = Some(op.clone()));
        out.borrow_mut()._parent_op = Some(op.clone());

        out
    }
}
