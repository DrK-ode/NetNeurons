use std::{f64::NAN, rc::Rc};

use super::*;

impl OpNode {
    fn check_size_and_shape(inp: &[TensorShared], same_shape: bool) {
        if !inp.is_empty() {
            let shape = inp[0].borrow()._shape;
            let length = inp[0].borrow()._value.len();
            if inp.iter().any(|t| {
                let s = t.borrow()._shape;
                s.0 * s.1 * s.2 == 0
                    || t.borrow()._value.is_empty()
                    || (same_shape && (s != shape || t.borrow()._value.len() != length))
            }) {
                panic!("Cannot operate on ill-sized tensor.");
            }
        } else {
            panic!("Cannot operate on nothing.");
        }
    }

    pub fn new_op(
        op: Box<dyn Operator>,
        inp: Vec<TensorShared>,
        same_input_shapes: bool,
    ) -> TensorShared {
        Self::check_size_and_shape(&inp, same_input_shapes);
        let out = TensorShared::from_shape(op.output_shape(&inp).unwrap());
        let op = Rc::new(OpNode {
            _op: op,
            _inp: inp,
            _out: out.clone(),
        });
        op._inp
            .iter()
            .for_each(|node| node.borrow_mut()._child_op = Some(op.clone()));
        op._out.borrow_mut()._parent_op = Some(op.clone());
        out
    }

    pub fn perform_operation(&self) {
        self._op.operate(&self._inp, &self._out)
    }

    pub fn back_propagate(&self) {
        self._op.back_propagate(&self._inp, &self._out);
    }
}

// Helper function for unary operators returning the same shape as input
fn operate_unary_same_shape<F: Fn(&FloatType) -> FloatType>(
    inp: &[TensorShared],
    out: &TensorShared,
    f: F,
) {
    inp[0]
        .borrow()
        ._value
        .iter()
        .zip(out.borrow_mut()._value.iter_mut())
        .for_each(|(inp, out)| *out = (f)(inp));
}

// Helper function for unary operators returning the same shape as input
fn unary_output_shape(input: &[TensorShared]) -> Option<TensorShape> {
    if input.len() == 1 {
        let input_shape = input[0].borrow()._shape;
        if input_shape.0 * input_shape.1 * input_shape.2 != 0 {
            Some(input_shape.to_owned())
        } else {
            None
        }
    } else {
        None
    }
}

// Helper function for unary operators returning same shape as input
fn back_propagate_unary_same_shape<F: Fn((FloatType, FloatType)) -> FloatType>(
    inp: &[TensorShared],
    out: &TensorShared,
    dfdx: F,
) {
    // Can I utilize std::men::take or RefCell::replace in order to zip and iterate?
    let inp = &inp[0];
    let size = inp.borrow()._value.len();
    for i in 0..size {
        let derivative = (dfdx)((inp.borrow()._value[i], out.borrow()._value[i]));
        inp.borrow_mut()._derivative[i] += derivative * out.borrow()._derivative[i];
    }
}

pub struct ExpOp {}
impl Operator for ExpOp {
    fn operate(&self, inp: &[TensorShared], out: &TensorShared) {
        operate_unary_same_shape(inp, out, |inp| inp.exp());
    }

    fn back_propagate(&self, inp: &[TensorShared], out: &TensorShared) {
        back_propagate_unary_same_shape(inp, out, |(_inp_val, out_val)| out_val)
    }

    fn symbol(&self) -> &str {
        "exp"
    }

    fn output_shape(&self, input: &[TensorShared]) -> Option<TensorShape> {
        unary_output_shape(input)
    }
}

pub struct LogOp {}
impl Operator for LogOp {
    fn operate(&self, inp: &[TensorShared], out: &TensorShared) {
        operate_unary_same_shape(inp, out, |inp| inp.ln());
    }

    fn back_propagate(&self, inp: &[TensorShared], out: &TensorShared) {
        back_propagate_unary_same_shape(inp, out, |(inp_val, _out_val)| 1.0 / inp_val)
    }

    fn symbol(&self) -> &str {
        "log"
    }

    fn output_shape(&self, input: &[TensorShared]) -> Option<TensorShape> {
        unary_output_shape(input)
    }
}

pub struct NegOp {}
impl Operator for NegOp {
    fn operate(&self, inp: &[TensorShared], out: &TensorShared) {
        operate_unary_same_shape(inp, out, |inp| -inp);
    }

    fn back_propagate(&self, inp: &[TensorShared], out: &TensorShared) {
        back_propagate_unary_same_shape(inp, out, |(_inp_val, _out_val)| -1.)
    }

    fn symbol(&self) -> &str {
        "¬"
    }

    fn output_shape(&self, input: &[TensorShared]) -> Option<TensorShape> {
        unary_output_shape(input)
    }
}

pub struct PowOp {}
impl Operator for PowOp {
    fn operate(&self, inp: &[TensorShared], out: &TensorShared) {
        let base = &inp[0].borrow()._value;
        let exp = &inp[1].borrow()._value;
        out.borrow_mut()._value = base
            .iter()
            .zip(exp)
            .map(|(base, exp)| base.powf(*exp))
            .collect();
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

    fn output_shape(&self, input: &[TensorShared]) -> Option<TensorShape> {
        if input.len() == 2 {
            let input_shape = input[0].borrow()._shape;
            if input_shape.0 * input_shape.1 * input_shape.2 != 0 {
                return Some(input_shape.to_owned());
            }
        }
        None
    }
}

pub struct AddOp {}
impl Operator for AddOp {
    fn operate(&self, inp: &[TensorShared], out: &TensorShared) {
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

    fn back_propagate(&self, inp: &[TensorShared], out: &TensorShared) {
        let size = inp[0].borrow()._value.len();
        for i in 0..size {
            let out_derivative = out.borrow()._derivative[i];
            for inp in inp {
                inp.borrow_mut()._derivative[i] += out_derivative;
            }
        }
    }

    fn symbol(&self) -> &str {
        "+"
    }

    fn output_shape(&self, input: &[TensorShared]) -> Option<TensorShape> {
        if input.len() >= 2 {
            let input_shape = input[0].borrow()._shape;
            if input_shape.0 * input_shape.1 * input_shape.2 != 0 {
                return Some(input_shape.to_owned());
            }
        }
        None
    }
}

pub struct SumOp {}
impl Operator for SumOp {
    fn operate(&self, inp: &[TensorShared], out: &TensorShared) {
        out.borrow_mut()._value = vec![inp[0].borrow()._value.iter().sum()];
    }

    fn back_propagate(&self, inp: &[TensorShared], out: &TensorShared) {
        let inp = &inp[0];
        let out_derivative = out.borrow()._derivative[0];
        inp.borrow_mut()
            ._derivative
            .iter_mut()
            .for_each(|derivative| *derivative += out_derivative);
    }

    fn symbol(&self) -> &str {
        "sum"
    }

    fn output_shape(&self, input: &[TensorShared]) -> Option<TensorShape> {
        if input.len() == 1 {
            let input_shape = input[0].borrow()._shape;
            if input_shape.0 * input_shape.1 * input_shape.2 != 0 {
                return Some((1, 1, 1));
            }
        }
        None
    }
}

pub struct MulOp {}
impl Operator for MulOp {
    fn operate(&self, inp: &[TensorShared], out: &TensorShared) {
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

    fn back_propagate(&self, inp: &[TensorShared], out: &TensorShared) {
        let size = inp[0].borrow()._value.len();
        for i in 0..size {
            let product = out.borrow()._value[i] * out.borrow()._derivative[i];
            for inp in inp {
                let derivative = product / inp.borrow()._value[i];
                inp.borrow_mut()._derivative[i] += derivative;
            }
        }
    }

    fn symbol(&self) -> &str {
        "*"
    }

    fn output_shape(&self, input: &[TensorShared]) -> Option<TensorShape> {
        if input.len() >= 2 {
            let input_shape = input[0].borrow()._shape;
            if input_shape.0 * input_shape.1 * input_shape.2 != 0 {
                return Some(input_shape.to_owned());
            }
        }
        None
    }
}

pub struct ProdOp {}
impl Operator for ProdOp {
    fn operate(&self, inp: &[TensorShared], out: &TensorShared) {
        out.borrow_mut()._value = vec![inp[0].borrow()._value.iter().product()];
    }

    fn back_propagate(&self, inp: &[TensorShared], out: &TensorShared) {
        let inp = &inp[0];
        let size = inp.borrow()._value.len();
        let product = out.borrow()._value[0] * out.borrow()._derivative[0];
        for i in 0..size {
            let derivative = product / inp.borrow()._value[i];
            inp.borrow_mut()._derivative[i] += derivative;
        }
    }

    fn symbol(&self) -> &str {
        "prod"
    }

    fn output_shape(&self, input: &[TensorShared]) -> Option<TensorShape> {
        if input.len() == 1 {
            let input_shape = input[0].borrow()._shape;
            if input_shape.0 * input_shape.1 * input_shape.2 != 0 {
                return Some((1, 1, 1));
            }
        }
        None
    }
}

pub struct DotOp {}
impl Operator for DotOp {
    fn operate(&self, inp: &[TensorShared], out: &TensorShared) {
        let lhs = &inp[0].borrow()._value;
        let rhs = &inp[1].borrow()._value;
        
        let lhs_rows = inp[0].shape().0;
        let lhs_cols = inp[0].shape().1;
        let rhs_rows = inp[1].shape().0;
        let rhs_cols = inp[1].shape().1;
        let out_rows = rhs_cols;
        let out_cols = lhs_rows;

        for (n, mat_elem) in out.borrow_mut()._value.iter_mut().enumerate() {
            let r = n / out_cols;
            let c = n % out_cols;
            let lhs_row = lhs.iter().skip(r * lhs_cols).take(lhs_cols);
            let rhs_col = rhs.iter().skip(c).step_by(rhs_cols).take(rhs_rows);
            println!("{}  {}",out_rows,lhs_row.clone().count());
            println!("{}  {}",out_cols,rhs_col.clone().count());
            *mat_elem = lhs_row.zip(rhs_col).map(|(&r, &c)| {
                println!("({}, {})",r,c);
                r * c
            }).sum();
        }
    }

    fn back_propagate(&self, inp: &[TensorShared], out: &TensorShared) {
        todo!()
    }

    fn symbol(&self) -> &str {
        "⋅"
    }

    fn output_shape(&self, input: &[TensorShared]) -> Option<TensorShape> {
        if input.len() == 2 {
            let shape1 = input[0].borrow()._shape;
            let shape2 = input[1].borrow()._shape;
            if shape1.0 > 0
                && shape1.1 > 0
                && shape1.2 == 1
                && shape2.0 == shape1.1
                && shape2.1 > 0
                && shape2.2 == 1
            {
                return Some((shape1.0, shape2.1, 1));
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn addition_of_two_scalars() {
        let inp1 = TensorShared::from_scalar(1.);
        let inp2 = TensorShared::from_scalar(2.);
        let out = &inp1 + &inp2;
        let calc = NetworkCalculation::new(&out);
        calc.forward();
        assert_eq!(out.value_as_scalar().unwrap(), 3.);
        calc.back_propagation();
        assert_eq!(out.derivative_as_scalar().unwrap(), 1.);
        assert_eq!(inp1.derivative_as_scalar().unwrap(), 1.);
        assert_eq!(inp2.derivative_as_scalar().unwrap(), 1.);
    }

    #[test]
    fn addition_of_scalar_to_itself() {
        let inp = TensorShared::from_scalar(1.);
        let out = &inp + &inp;
        let calc = NetworkCalculation::new(&out);
        calc.forward();
        assert_eq!(out.value_as_scalar().unwrap(), 2.);
        calc.back_propagation();
        assert_eq!(out.derivative_as_scalar().unwrap(), 1.);
        assert_eq!(inp.derivative_as_scalar().unwrap(), 2.);
    }

    #[test]
    fn addition_of_two_vectors() {
        let inp1 = TensorShared::from_vector(vec![1., 2.], (2, 1, 1));
        let inp2 = TensorShared::from_vector(vec![3., 4.], (2, 1, 1));
        let expected_value = vec![4., 6.];
        let expected_derivative1 = vec![1., 1.];
        let expected_derivative2 = vec![1., 1.];
        let out = &inp1 + &inp2;
        let calc = NetworkCalculation::new(&out);
        calc.forward();
        assert_eq!(out.value_as_col_vector().unwrap(), expected_value);
        calc.back_propagation();
        assert_eq!(out.derivative(), vec![1., 1.]);
        assert_eq!(inp1.derivative(), expected_derivative1);
        assert_eq!(inp2.derivative(), expected_derivative2);
    }

    #[test]
    fn sum_of_tensor_elements() {
        let inp = TensorShared::from_vector(vec![1., 2., 3., 4.], (1, 2, 2));
        let expected_value = 10.;
        let expected_derivative = vec![1., 1., 1., 1.];
        let out = inp.sum();
        let calc = NetworkCalculation::new(&out);
        calc.forward();
        assert_eq!(out.value_as_scalar().unwrap(), expected_value);
        calc.back_propagation();
        assert_eq!(out.derivative(), vec![1.]);
        assert_eq!(inp.derivative(), expected_derivative);
    }

    #[test]
    fn product_of_two_vectors() {
        let inp1 = TensorShared::from_vector(vec![1., 2.], (2, 1, 1));
        let inp2 = TensorShared::from_vector(vec![3., 4.], (2, 1, 1));
        let expected_value = vec![3., 8.];
        let expected_derivative1 = vec![3., 4.];
        let expected_derivative2 = vec![1., 2.];
        let out = &inp1 * &inp2;
        let calc = NetworkCalculation::new(&out);
        calc.forward();
        assert_eq!(out.value_as_col_vector().unwrap(), expected_value);
        calc.back_propagation();
        assert_eq!(out.derivative(), vec![1., 1.]);
        assert_eq!(inp1.derivative(), expected_derivative1);
        assert_eq!(inp2.derivative(), expected_derivative2);
    }

    #[test]
    fn product_of_tensor_elements() {
        let inp = TensorShared::from_vector(vec![1., 2., 3., 4.], (1, 2, 2));
        let expected_value = 24.;
        let expected_derivative = vec![24., 12., 8., 6.];
        let out = inp.product();
        let calc = NetworkCalculation::new(&out);
        calc.forward();
        assert_eq!(out.value_as_scalar().unwrap(), expected_value);
        calc.back_propagation();
        assert_eq!(out.derivative(), vec![1.]);
        assert_eq!(inp.derivative(), expected_derivative);
    }

    #[test]
    fn power_of_two_vectors() {
        let inp1 = TensorShared::from_vector(vec![1., 2.], (2, 1, 1));
        let inp2 = TensorShared::from_vector(vec![3., 4.], (2, 1, 1));
        let expected_value = vec![1., 16.];
        let expected_derivative1 = vec![3., 32.];
        let expected_derivative2 = vec![0., (2 as FloatType).ln() * 16.];
        let out = inp1.pow(&inp2);
        let calc = NetworkCalculation::new(&out);
        calc.forward();
        assert_eq!(out.value(), expected_value);
        calc.back_propagation();
        assert_eq!(out.derivative(), vec![1., 1.]);
        assert_eq!(inp1.derivative(), expected_derivative1);
        assert_eq!(inp2.derivative(), expected_derivative2);
    }

    #[test]
    fn negation_of_tensor() {
        let inp: Vec<FloatType> = vec![1., 2., 3., 4.];
        let expected_value = inp.iter().map(|val| -val).collect::<Vec<_>>();
        let expected_derivative = inp.iter().map(|_| -1.).collect::<Vec<_>>();
        let inp = TensorShared::from_vector(inp, (1, 2, 2));
        let out = -(&inp);
        let calc = NetworkCalculation::new(&out);
        calc.forward();
        assert_eq!(out.value(), expected_value);
        calc.back_propagation();
        assert_eq!(out.derivative(), vec![1., 1., 1., 1.]);
        assert_eq!(inp.derivative(), expected_derivative);
    }

    #[test]
    fn log_of_tensor() {
        let inp: Vec<FloatType> = vec![1., 2., 3., 4.];
        let expected_value = inp.iter().map(|val| val.ln()).collect::<Vec<_>>();
        let expected_derivative = inp.iter().map(|val| 1. / val).collect::<Vec<_>>();
        let inp = TensorShared::from_vector(inp, (1, 2, 2));
        let out = inp.log();
        let calc = NetworkCalculation::new(&out);
        calc.forward();
        assert_eq!(out.value(), expected_value);
        calc.back_propagation();
        assert_eq!(out.derivative(), vec![1., 1., 1., 1.]);
        assert_eq!(inp.derivative(), expected_derivative);
    }

    #[test]
    fn exp_of_tensor() {
        let inp: Vec<FloatType> = vec![1., 2., 3., 4.];
        let expected_value = inp.iter().map(|val| val.exp()).collect::<Vec<_>>();
        let expected_derivative = inp.iter().map(|val| val.exp()).collect::<Vec<_>>();
        let inp = TensorShared::from_vector(inp, (1, 2, 2));
        let out = inp.exp();
        let calc = NetworkCalculation::new(&out);
        calc.forward();
        assert_eq!(out.value(), expected_value);
        calc.back_propagation();
        assert_eq!(out.derivative(), vec![1., 1., 1., 1.]);
        assert_eq!(inp.derivative(), expected_derivative);
    }

    #[test]
    fn long_expression() {
        let a = TensorShared::from_scalar(1.0);
        let b = TensorShared::from_scalar(-1.0);
        let c = TensorShared::from_scalar(1.0);
        let c2 = TensorShared::from_scalar(2.0);
        let c3 = TensorShared::from_scalar(3.0);
        let d = &a + &b - (&c * &c2);
        let e = (-&d).pow(&c3);
        let f = e.log();
        let g = f.exp();

        let calc = NetworkCalculation::new(&g);
        calc.forward();
        assert_approx_eq!(d.value_as_scalar().unwrap(), -2.);
        assert_approx_eq!(e.value_as_scalar().unwrap(), 8.);
        assert_approx_eq!(f.value_as_scalar().unwrap(), 8f64.ln());
        assert_approx_eq!(g.value_as_scalar().unwrap(), 8.);
        calc.back_propagation();
        assert_approx_eq!(g.derivative_as_scalar().unwrap(), 1.);
        assert_approx_eq!(f.derivative_as_scalar().unwrap(), 8.);
        assert_approx_eq!(e.derivative_as_scalar().unwrap(), 1.);
        assert_approx_eq!(d.derivative_as_scalar().unwrap(), -12.);
        assert_approx_eq!(c.derivative_as_scalar().unwrap(), 24.);
        assert_approx_eq!(b.derivative_as_scalar().unwrap(), -12.);
        assert_approx_eq!(a.derivative_as_scalar().unwrap(), -12.);
    }

    #[test]
    fn dot_product_of_two_vectors() {
        let inp1 = TensorShared::from_vector(vec![1., 2.], (1, 2, 1));
        let inp2 = TensorShared::from_vector(vec![3., 4.], (2, 1, 1));
        let expected_value = 11.;
        let expected_derivative1 = vec![3., 4.];
        let expected_derivative2 = vec![1., 2.];
        let out = inp1.dot(&inp2);
        let calc = NetworkCalculation::new(&out);
        calc.forward();
        assert_eq!(out.value_as_scalar().unwrap(), expected_value);
        calc.back_propagation();
        assert_eq!(out.derivative(), vec![1., 1.]);
        assert_eq!(inp1.derivative(), expected_derivative1);
        assert_eq!(inp2.derivative(), expected_derivative2);
    }
}
