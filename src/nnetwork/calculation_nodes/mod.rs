use std::{cell::RefCell, fmt::Debug, rc::Rc};

type FloatType = f64;
pub type TensorShared = Rc<RefCell<Tensor>>;
pub type OpNodeShared = Rc<OpNode>;

#[derive(Default)]
pub struct Tensor {
    _parent_op: Option<OpNodeShared>,
    _child_op: Option<OpNodeShared>,
    _shape: (usize, usize, usize),
    _values: Vec<FloatType>,
    _derivative: Vec<FloatType>,
}

impl Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field(
                "_parent_op",
                &(match self._parent_op.as_ref() {
                    Some(op) => format!("Some({})", op._op),
                    None => "None".to_string(),
                })
                .to_string(),
            )
            .field(
                "_child_op",
                &(match self._child_op.as_ref() {
                    Some(op) => format!("Some({})", op._op),
                    None => "None".to_string(),
                })
                .to_string(),
            )
            .field("_shape", &self._shape)
            .field("_values", &self._values)
            .field("_derivative", &self._derivative)
            .finish()
    }
}

#[derive(Debug)]
pub enum NodeOp {
    Exp,
    Log,
    Neg,
    Add,
    Mul,
    Pow,
}

#[derive(Debug)]
pub struct OpNode {
    _op: NodeOp,
    _inp: Vec<TensorShared>,
    _out: TensorShared,
}

mod network_calculation;
mod op_node;
mod tensor;
