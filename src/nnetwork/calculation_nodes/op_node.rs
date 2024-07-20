use std::{f64::NAN, rc::Rc};

use super::*;

impl OpNode {
    fn setup_same_shape_unary_op(op: Box<dyn Operator>, inp: &TensorShared) -> TensorShared {
        let inp_vec = vec![inp.clone()];
        Self::check_size(&inp_vec);
        let out_vec = vec![NAN; inp.borrow()._value.len()];
        let out = Tensor::from_vector(out_vec, inp.borrow()._shape);
        Self::setup_op(op, vec![inp.clone()], out.clone());
        out
    }

    fn setup_same_shape_multi_op(op: Box<dyn Operator>, inp: &[TensorShared]) -> TensorShared {
        let (length, shape) = Self::check_same_shape(inp);
        let out = Tensor::from_vector(vec![NAN; length], shape);
        Self::setup_op(op, inp.into(), out.clone());
        out
    }

    fn setup_op(op: Box<dyn Operator>, inp: Vec<TensorShared>, out: TensorShared) {
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
            x == 0 || y == 0 || z == 0 || t.borrow()._value.is_empty()
        }) {
            panic!("Cannot operate on null-sized tensor.");
        }
    }

    fn check_same_shape(inp: &[TensorShared]) -> (usize, TensorShape) {
        let length = inp[0].borrow()._value.len();
        let shape = inp[0].borrow()._shape;
        if inp
            .iter()
            .any(|t| t.borrow()._shape != shape || t.borrow()._value.len() != length)
        {
            panic!("Not all input tensors are of the same shape and size.");
        }
        (length, shape)
    }

    pub fn exp(inp: &TensorShared) -> TensorShared {
        // Exp is a unary operator so it will always be performed element-wise
        Self::setup_same_shape_unary_op(Box::new(ExpOp {}), inp)
    }
    pub fn log(inp: &TensorShared) -> TensorShared {
        // Log is a unary operator so it will always be performed element-wise
        Self::setup_same_shape_unary_op(Box::new(LogOp {}), inp)
    }
    pub fn neg(inp: &TensorShared) -> TensorShared {
        // Neg is a unary operator so it will always be performed element-wise
        Self::setup_same_shape_unary_op(Box::new(NegOp {}), inp)
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
        Self::setup_same_shape_multi_op(Box::new(AddOp {}), inp)
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
        Self::setup_same_shape_multi_op(Box::new(MulOp {}), inp)
    }
    pub fn pow(inp1: &TensorShared, inp2: &TensorShared) -> TensorShared {
        // Pow is an element-wise two-input operator. Operands must be of the same shape.
        let inp = vec![inp1.clone(), inp2.clone()];
        Self::setup_same_shape_multi_op(Box::new(PowOp {}), &inp)
    }
    pub fn sum(inp: &TensorShared) -> TensorShared {
        // Sum adds all elements in a tensor together. Implemented as a variant of Add.
        let inp = vec![inp.clone()];
        let out = Tensor::from_shape((1, 1, 1));
        Self::check_size(&inp);
        Self::setup_op(Box::new(AddOp {}), inp, out.clone());
        out
    }
    pub fn prod(inp: &TensorShared) -> TensorShared {
        // Prod multiplies all elements in a tensor together. Implemented as a unary variant of Mul.
        let inp = vec![inp.clone()];
        let out = Tensor::from_shape((1, 1, 1));
        Self::check_size(&inp);
        Self::setup_op(Box::new(MulOp {}), inp, out.clone());
        out
    }

    pub fn perform_operation(&self) {
        self._op.operate(&self._inp, &self._out)
    }

    pub fn back_propagate(&self) {
        self._op.back_propagate(&self._inp, &self._out);
    }
}

// Helper function for unary operators returning same shape as input
fn operate_unary_same_shape<F: FnMut((&FloatType, &mut FloatType))>(
    inp: &[TensorShared],
    out: &TensorShared,
    f: F,
) {
    inp[0]
        .borrow()
        ._value
        .iter()
        .zip(out.borrow_mut()._value.iter_mut())
        .for_each(f);
}

// Helper function for unary operators returning same shape as input
fn back_propagate_unary_same_shape<F: Fn((FloatType, (FloatType, FloatType))) -> FloatType>(
    inp: &[TensorShared],
    out: &TensorShared,
    dfdx: F,
) {
    let inp = &inp[0];
    let size = inp.borrow()._value.len();
    for i in 0..size {
        let derivative = (dfdx)((
            inp.borrow()._value[i],
            (out.borrow()._value[i], out.borrow()._derivative[i]),
        ));
        inp.borrow_mut()._derivative[i] += derivative;
    }
}

struct ExpOp {}
impl Operator for ExpOp {
    fn operate(&self, inp: &[TensorShared], out: &TensorShared) {
        match inp.len() {
            1 => {
                operate_unary_same_shape(inp, out, |(inp, out)| *out = inp.exp());
            }
            _ => panic!("Exponentiation is an exclusive unary operator."),
        }
    }

    fn back_propagate(&self, inp: &[TensorShared], out: &TensorShared) {
        back_propagate_unary_same_shape(inp, out, |(_inp_val, (out_val, chain_derivative))| {
            out_val * chain_derivative
        })
    }

    fn symbol(&self) -> &str {
        "exp"
    }
}

struct LogOp {}
impl Operator for LogOp {
    fn operate(&self, inp: &[TensorShared], out: &TensorShared) {
        match inp.len() {
            1 => out
                .borrow_mut()
                ._value
                .iter_mut()
                .zip(inp[0].borrow()._value.iter())
                .for_each(|(out, a)| *out = a.ln()),

            _ => panic!("Logarithmation is an exclusive unary operator."),
        }
    }

    fn back_propagate(&self, inp: &[TensorShared], out: &TensorShared) {
        back_propagate_unary_same_shape(inp, out, |(inp_val, (_out_val, chain_derivative))| {
            chain_derivative / inp_val
        })
    }

    fn symbol(&self) -> &str {
        "log"
    }
}

struct NegOp {}
impl Operator for NegOp {
    fn operate(&self, inp: &[TensorShared], out: &TensorShared) {
        match inp.len() {
            1 => out
                .borrow_mut()
                ._value
                .iter_mut()
                .zip(inp[0].borrow()._value.iter())
                .for_each(|(out, a)| *out = -*a),
            _ => panic!("Negation is an exclusive unary operator."),
        }
    }

    fn back_propagate(&self, inp: &[TensorShared], out: &TensorShared) {
        back_propagate_unary_same_shape(inp, out, |(_, (_, _))| -1.)
    }

    fn symbol(&self) -> &str {
        "¬"
    }
}

struct PowOp {}
impl Operator for PowOp {
    fn operate(&self, inp: &[TensorShared], out: &TensorShared) {
        match inp.len() {
            2 => {
                let base = &inp[0].borrow()._value;
                let exp = &inp[1].borrow()._value;
                out.borrow_mut()._value = base
                    .iter()
                    .zip(exp)
                    .map(|(base, exp)| base.powf(*exp))
                    .collect();
            }
            _ => panic!("Pow operator cannot operate on any other number of arguments than two."),
        }
    }

    fn back_propagate(&self, inp: &[TensorShared], out: &TensorShared) {
        let base = &inp[0];
        let exp = &inp[1];
        let size = base.borrow()._value.len();
        for i in 0..size {
            let exp_value = exp.borrow()._value[i];
            let base_value = base.borrow()._value[i];
            let out_value = out.borrow()._value[i];
            let out_derivative = out.borrow()._derivative[i];
            let base_derivative = exp_value * base_value.powf(exp_value - 1.) * out_derivative;
            let exp_derivative = base_value.ln() * out_value * out_derivative;
            base.borrow_mut()._derivative[i] += base_derivative;
            exp.borrow_mut()._derivative[i] += exp_derivative;
        }
    }

    fn symbol(&self) -> &str {
        "^"
    }
}

struct AddOp {}
impl Operator for AddOp {
    fn operate(&self, inp: &[TensorShared], out: &TensorShared) {
        match inp.len() {
            0 => panic!("Cannot operate on zero operands."),
            1 => out.borrow_mut()._value = vec![inp[0].borrow()._value.iter().sum()],
            _ => {
                let mut out_vec = vec![0.; inp[0].borrow()._value.len()];
                inp.iter().for_each(|node| {
                    let vec = &node.borrow()._value;
                    out_vec
                        .iter_mut()
                        .zip(vec)
                        .for_each(|(out, inp)| *out += *inp);
                });
                out.borrow_mut()._value = out_vec;
            }
        }
    }

    fn back_propagate(&self, inp: &[TensorShared], out: &TensorShared) {
        match inp.len() {
            1 => {
                let inp = &inp[0];
                let out_derivative = out.borrow()._derivative[0];
                inp.borrow_mut()
                    ._derivative
                    .iter_mut()
                    .for_each(|derivative| *derivative += out_derivative);
            }
            _ => {
                let size = inp[0].borrow()._value.len();
                for i in 0..size {
                    let out_derivative = out.borrow()._derivative[i];
                    for inp in inp {
                        inp.borrow_mut()._derivative[i] += out_derivative;
                    }
                }
            }
        }
    }

    fn symbol(&self) -> &str {
        "+"
    }
}

struct MulOp {}
impl Operator for MulOp {
    fn operate(&self, inp: &[TensorShared], out: &TensorShared) {
        match inp.len() {
            0 => panic!("Cannot operate on zero operands."),
            1 => out.borrow_mut()._value = vec![inp[0].borrow()._value.iter().product()],
            _ => {
                out.borrow_mut()._value.iter_mut().for_each(|val| *val = 1.);
                inp.iter().for_each(|node| {
                    let vec = &node.borrow()._value;
                    out.borrow_mut()
                        ._value
                        .iter_mut()
                        .zip(vec)
                        .for_each(|(out, inp)| *out *= *inp);
                });
            }
        }
    }

    fn back_propagate(&self, inp: &[TensorShared], out: &TensorShared) {
        match inp.len() {
            1 => {
                let inp = &inp[0];
                let size = inp.borrow()._value.len();
                let product = out.borrow()._value[0] * out.borrow()._derivative[0];
                for i in 0..size {
                    let derivative = product / inp.borrow()._value[i];
                    inp.borrow_mut()._derivative[i] += derivative;
                }
            }
            _ => {
                let size = inp[0].borrow()._value.len();
                for i in 0..size {
                    let product = out.borrow()._value[i] * out.borrow()._derivative[i];
                    for inp in inp {
                        let derivative = product / inp.borrow()._value[i];
                        inp.borrow_mut()._derivative[i] += derivative;
                    }
                }
            }
        }
    }

    fn symbol(&self) -> &str {
        "*"
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
        println!("{:?}", out.borrow()._value);
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
        let inp1 = Tensor::from_vector(vec![1., 2.], (2, 1, 1));
        let inp2 = Tensor::from_vector(vec![3., 4.], (2, 1, 1));
        let expected_value = vec![4., 6.];
        let expected_derivative1 = vec![1., 1.];
        let expected_derivative2 = vec![1., 1.];
        let out = OpNode::add(&inp1, &inp2);
        let calc = NetworkCalculation::new(&out);
        calc.forward();
        assert_eq!(out.borrow().as_col_vector().unwrap(), expected_value);
        calc.back_propagation();
        assert_eq!(out.borrow().derivative(), vec![1., 1.]);
        assert_eq!(inp1.borrow().derivative(), expected_derivative1);
        assert_eq!(inp2.borrow().derivative(), expected_derivative2);
    }

    #[test]
    fn sum_of_tensor_elements() {
        let inp = Tensor::from_vector(vec![1., 2., 3., 4.], (1, 2, 2));
        let expected_value = 10.;
        let expected_derivative = vec![1., 1., 1., 1.];
        let out = OpNode::sum(&inp);
        let calc = NetworkCalculation::new(&out);
        calc.forward();
        assert_eq!(out.borrow().as_scalar().unwrap(), expected_value);
        calc.back_propagation();
        assert_eq!(out.borrow().derivative(), vec![1.]);
        assert_eq!(inp.borrow().derivative(), expected_derivative);
    }

    #[test]
    fn product_of_two_vectors() {
        let inp1 = Tensor::from_vector(vec![1., 2.], (2, 1, 1));
        let inp2 = Tensor::from_vector(vec![3., 4.], (2, 1, 1));
        let expected_value = vec![3., 8.];
        let expected_derivative1 = vec![3., 4.];
        let expected_derivative2 = vec![1., 2.];
        let out = OpNode::mul(&inp1, &inp2);
        let calc = NetworkCalculation::new(&out);
        calc.forward();
        assert_eq!(out.borrow().as_col_vector().unwrap(), expected_value);
        calc.back_propagation();
        assert_eq!(out.borrow().derivative(), vec![1., 1.]);
        assert_eq!(inp1.borrow().derivative(), expected_derivative1);
        assert_eq!(inp2.borrow().derivative(), expected_derivative2);
    }

    #[test]
    fn product_of_tensor_elements() {
        let inp = Tensor::from_vector(vec![1., 2., 3., 4.], (1, 2, 2));
        let expected_value = 24.;
        let expected_derivative = vec![24., 12., 8., 6.];
        let out = OpNode::prod(&inp);
        let calc = NetworkCalculation::new(&out);
        calc.forward();
        assert_eq!(out.borrow().as_scalar().unwrap(), expected_value);
        calc.back_propagation();
        assert_eq!(out.borrow().derivative(), vec![1.]);
        assert_eq!(inp.borrow().derivative(), expected_derivative);
    }

    #[test]
    fn power_of_two_vectors() {
        let inp1 = Tensor::from_vector(vec![1., 2.], (2, 1, 1));
        let inp2 = Tensor::from_vector(vec![3., 4.], (2, 1, 1));
        let expected_value = vec![1., 16.];
        let expected_derivative1 = vec![3., 32.];
        let expected_derivative2 = vec![0., (2 as FloatType).ln() * 16.];
        let out = OpNode::pow(&inp1, &inp2);
        let calc = NetworkCalculation::new(&out);
        calc.forward();
        assert_eq!(out.borrow().data(), expected_value);
        calc.back_propagation();
        assert_eq!(out.borrow().derivative(), vec![1., 1.]);
        assert_eq!(inp1.borrow().derivative(), expected_derivative1);
        assert_eq!(inp2.borrow().derivative(), expected_derivative2);
    }

    #[test]
    fn negation_of_tensor() {
        let inp: Vec<FloatType> = vec![1., 2., 3., 4.];
        let expected_value = inp.iter().map(|val| -val).collect::<Vec<_>>();
        let expected_derivative = inp.iter().map(|_| -1.).collect::<Vec<_>>();
        let inp = Tensor::from_vector(inp, (1, 2, 2));
        let out = OpNode::neg(&inp);
        let calc = NetworkCalculation::new(&out);
        calc.forward();
        assert_eq!(out.borrow().data(), expected_value);
        calc.back_propagation();
        assert_eq!(out.borrow().derivative(), vec![1., 1., 1., 1.]);
        assert_eq!(inp.borrow().derivative(), expected_derivative);
    }

    #[test]
    fn log_of_tensor() {
        let inp: Vec<FloatType> = vec![1., 2., 3., 4.];
        let expected_value = inp.iter().map(|val| val.ln()).collect::<Vec<_>>();
        let expected_derivative = inp.iter().map(|val| 1. / val).collect::<Vec<_>>();
        let inp = Tensor::from_vector(inp, (1, 2, 2));
        let out = OpNode::log(&inp);
        let calc = NetworkCalculation::new(&out);
        calc.forward();
        assert_eq!(out.borrow().data(), expected_value);
        calc.back_propagation();
        assert_eq!(out.borrow().derivative(), vec![1., 1., 1., 1.]);
        assert_eq!(inp.borrow().derivative(), expected_derivative);
    }

    #[test]
    fn exp_of_tensor() {
        let inp: Vec<FloatType> = vec![1., 2., 3., 4.];
        let expected_value = inp.iter().map(|val| val.exp()).collect::<Vec<_>>();
        let expected_derivative = inp.iter().map(|val| val.exp()).collect::<Vec<_>>();
        let inp = Tensor::from_vector(inp, (1, 2, 2));
        let out = OpNode::exp(&inp);
        let calc = NetworkCalculation::new(&out);
        calc.forward();
        assert_eq!(out.borrow().data(), expected_value);
        calc.back_propagation();
        assert_eq!(out.borrow().derivative(), vec![1., 1., 1., 1.]);
        assert_eq!(inp.borrow().derivative(), expected_derivative);
    }
}
