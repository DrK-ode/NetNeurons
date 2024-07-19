use std::{f64::NAN, fmt::Display, rc::Rc};

use super::*;

impl Display for NodeOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                NodeOp::Exp => "exp",
                NodeOp::Log => "log",
                NodeOp::Neg => "-",
                NodeOp::Add => "+",
                NodeOp::Mul => "*",
                NodeOp::Pow => "^",
            }
        )
    }
}

impl OpNode {
    fn setup_same_shape_unary_op(op: NodeOp, inp: &TensorShared) -> TensorShared {
        let inp_vec = vec![inp.clone()];
        Self::check_size(&inp_vec);
        let out_vec = vec![NAN; inp.borrow()._values.len()];
        let out = Tensor::from_vector(out_vec, inp.borrow()._shape);
        Self::setup_op(op, vec![inp.clone()], out.clone());
        out
    }

    fn setup_same_shape_multi_op(op: NodeOp, inp: &[TensorShared]) -> TensorShared {
        let (length, shape) = Self::check_same_shape(inp);
        let out = Tensor::from_vector(vec![NAN; length], shape);
        Self::setup_op(op, inp.into(), out.clone());
        out
    }

    fn setup_op(op: NodeOp, inp: Vec<TensorShared>, out: TensorShared) {
        let op = Rc::new(OpNode {
            _op: op,
            _inp: inp,
            _out: out,
        });
        op._inp
            .iter()
            .for_each(|node| node.borrow_mut()._child_op = Some(op.clone()));
        op._out.borrow_mut()._parent_op = Some(op.clone());
    }

    fn check_size(inp: &[TensorShared]) {
        if inp.iter().any(|t| {
            let (x, y, z) = t.borrow()._shape;
            x == 0 || y == 0 || z == 0 || t.borrow()._values.len() == 0
        }) {
            panic!("Cannot operate on null-sized tensor.");
        }
    }

    fn check_same_shape(inp: &[TensorShared]) -> (usize, (usize, usize, usize)) {
        let length = inp[0].borrow()._values.len();
        let shape = inp[0].borrow()._shape;
        if inp
            .iter()
            .any(|t| t.borrow()._shape != shape || t.borrow()._values.len() != length)
        {
            panic!("Not all input tensors are of the same shape and size.");
        }
        (length, shape)
    }

    pub fn exp(inp: &TensorShared) -> TensorShared {
        // Exp is a unary operator so it will always be performed element-wise
        Self::setup_same_shape_unary_op(NodeOp::Exp, inp)
    }
    pub fn log(inp: &TensorShared) -> TensorShared {
        // Log is a unary operator so it will always be performed element-wise
        Self::setup_same_shape_unary_op(NodeOp::Log, inp)
    }
    pub fn neg(inp: &TensorShared) -> TensorShared {
        // Neg is a unary operator so it will always be performed element-wise
        Self::setup_same_shape_unary_op(NodeOp::Neg, inp)
    }
    pub fn add(inp1: &TensorShared, inp2: &TensorShared) -> TensorShared {
        // Add is a multi input operator. Operands must be of the same shape.
        let inp = vec![inp1.clone(), inp2.clone()];
        Self::add_multi(&inp)
    }
    pub fn add_multi(inp: &[TensorShared]) -> TensorShared {
        // Add is a multi input operator. Operands must be of the same shape.
        if inp.len() == 0 {
            panic!("Cannot operate on zero operands.");
        }
        Self::setup_same_shape_multi_op(NodeOp::Add, inp)
    }
    pub fn mul(inp1: &TensorShared, inp2: &TensorShared) -> TensorShared {
        let inp = vec![inp1.clone(), inp2.clone()];
        Self::mul_multi(&inp)
    }
    pub fn mul_multi(inp: &[TensorShared]) -> TensorShared {
        // Mul is an element-wise multi input operator. Operands must be of the same shape.
        if inp.len() == 0 {
            panic!("Cannot operate on zero operands.");
        }
        Self::setup_same_shape_multi_op(NodeOp::Mul, inp)
    }
    pub fn pow(inp1: &TensorShared, inp2: &TensorShared) -> TensorShared {
        // Pow is an element-wise two-input operator. Operands must be of the same shape.
        let inp = vec![inp1.clone(), inp2.clone()];
        Self::setup_same_shape_multi_op(NodeOp::Mul, &inp)
    }
    pub fn sum(inp: &TensorShared) -> TensorShared {
        // Sum adds all elements in a tensor together. Implemented as a variant of Add.
        let inp = vec![inp.clone()];
        let out = Tensor::from_shape((1,1,1));
        Self::check_size(&inp);
        Self::setup_op(NodeOp::Add, inp, out.clone());
        out
    }
    pub fn prod(inp: &TensorShared) -> TensorShared {
        // Prod multiplies all elements in a tensor together. Implemented as a unary variant of Mul.
        Self::add_multi(&vec![inp.clone()])
    }

    fn perform_operation(&self) {
        match &self._op {
            NodeOp::Exp => Self::perform_exp(&self._inp, &self._out),
            NodeOp::Log => Self::perform_log(&self._inp, &self._out),
            NodeOp::Neg => Self::perform_neg(&self._inp, &self._out),
            NodeOp::Add => Self::perform_add(&self._inp, &self._out),
            NodeOp::Mul => Self::perform_mul(&self._inp, &self._out),
            NodeOp::Pow => Self::perform_pow(&self._inp, &self._out),
        }
    }

    fn perform_exp(inp: &[TensorShared], out: &TensorShared) {
        match inp.len() {
            1 => {
                out.borrow_mut()._values = inp[0].borrow()._values.iter().map(|a| a.exp()).collect()
            }
            _ => panic!("Exponentiation is an exclusive unary operator."),
        }
    }

    fn perform_log(inp: &[TensorShared], out: &TensorShared) {
        match inp.len() {
            1 => {
                out.borrow_mut()._values = inp[0].borrow()._values.iter().map(|a| a.ln()).collect()
            }
            _ => panic!("Logarithmation is an exclusive unary operator."),
        }
    }

    fn perform_neg(inp: &[TensorShared], out: &TensorShared) {
        match inp.len() {
            1 => out.borrow_mut()._values = inp[0].borrow()._values.iter().map(|a| -a).collect(),
            _ => panic!("Negation is an exclusive unary operator."),
        }
    }

    fn perform_pow(inp: &[TensorShared], out: &TensorShared) {
        match inp.len() {
            2 => {
                let base = &inp[0].borrow()._values;
                let exp = &inp[0].borrow()._values;
                out.borrow_mut()._values = base
                    .iter()
                    .zip(exp)
                    .map(|(base, exp)| base.powf(*exp))
                    .collect();
            }
            _ => panic!("Pow operator cannot operate on any other number of arguments than two."),
        }
    }

    fn perform_add(inp: &[TensorShared], out: &TensorShared) {
        match inp.len() {
            0 => panic!("Cannot operate on zero operands."),
            1 => {
                println!("{:?}", inp);
                out.borrow_mut()._values = vec![inp[0].borrow()._values.iter().sum()]
            }
            _ => {
                let mut out_vec = vec![0.; inp[0].borrow()._values.len()];
                inp.iter().for_each(|node| {
                    let vec = &node.borrow()._values;
                    out_vec
                        .iter_mut()
                        .zip(vec)
                        .for_each(|(out, inp)| *out += *inp);
                });
                out.borrow_mut()._values = out_vec;
            }
        }
    }

    fn perform_mul(inp: &[TensorShared], out: &TensorShared) {
        match inp.len() {
            0 => panic!("Cannot operate on zero operands."),
            1 => out.borrow_mut()._values = vec![inp[0].borrow()._values.iter().product()],
            _ => {
                let mut out_vec = vec![0.; inp[0].borrow()._values.len()];
                inp.iter().for_each(|node| {
                    let vec = &node.borrow()._values;
                    out_vec
                        .iter_mut()
                        .zip(vec)
                        .for_each(|(out, inp)| *out *= *inp);
                });
                out.borrow_mut()._values = out_vec;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn addition_of_two_scalars() {
        let inp1 = Tensor::from_scalar(1.);
        let inp2 = Tensor::from_scalar(2.);
        let out = OpNode::add(&inp1, &inp2);
        inp1.borrow()
            ._child_op
            .as_ref()
            .unwrap()
            .perform_operation();
        println!("{:?}", out.borrow()._values);
        assert_eq!(out.borrow().as_scalar().unwrap(), 3.);
    }

    #[test]
    fn addition_of_scalar_to_itself() {
        let inp1 = Tensor::from_scalar(1.);
        let out = OpNode::add(&inp1, &inp1);
        inp1.borrow()
            ._child_op
            .as_ref()
            .unwrap()
            .perform_operation();
        assert_eq!(out.borrow().as_scalar().unwrap(), 2.);
    }

    #[test]
    fn addition_of_two_vectors() {
        let inp1 = Tensor::from_vector(vec![1., 2.], (1, 2, 1));
        let inp2 = Tensor::from_vector(vec![3., 4.], (1, 2, 1));
        let out = OpNode::add(&inp1, &inp2);
        inp1.borrow()
            ._child_op
            .as_ref()
            .unwrap()
            .perform_operation();
        assert_eq!(out.borrow().as_row_vector().unwrap(), vec![4., 6.]);
    }

    #[test]
    fn summing_a_tensor() {
        let inp = Tensor::from_vector(vec![1., 2., 3., 4.], (1, 2, 2));
        let out = OpNode::sum(&inp);
        inp.borrow()._child_op.as_ref().unwrap().perform_operation();
        assert_eq!(out.borrow().as_scalar().unwrap(), 10.);
    }
}
