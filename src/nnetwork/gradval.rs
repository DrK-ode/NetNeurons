use std::{
    cell::RefCell,
    ops::{Add, Div, Mul, Neg, Sub},
    rc::Rc,
};

type Ancestor = Rc<RefCell<Gv>>;

#[derive(Clone, Debug, Default, PartialEq)]
enum GradValOp {
    #[default]
    Noop,
    Neg(Ancestor),
    Exp(Ancestor),
    Log(Ancestor),
    Pow(Ancestor, Ancestor),
    Add(Ancestor, Ancestor),
    Sub(Ancestor, Ancestor),
    Mul(Ancestor, Ancestor),
    Div(Ancestor, Ancestor),
}

#[derive(Debug, Default)]
struct Gv {
    _val: f32,
    _grad: Option<f32>,
    _op: GradValOp,
}

impl Gv {
    fn from_value(v: f32) -> Self {
        Gv {
            _val: v,
            ..Self::default()
        }
    }

    fn from_op(v: f32, op: GradValOp) -> Self {
        Gv {
            _val: v,
            _op: op,
            ..Self::default()
        }
    }

    fn reset_grad_recursively(&mut self) {
        self._grad = None;
        match &self._op {
            GradValOp::Noop => {}
            GradValOp::Neg(a) => a.borrow_mut().reset_grad_recursively(),
            GradValOp::Exp(a) => a.borrow_mut().reset_grad_recursively(),
            GradValOp::Log(a) => a.borrow_mut().reset_grad_recursively(),
            GradValOp::Pow(a, b) => {
                a.borrow_mut().reset_grad_recursively();
                b.borrow_mut().reset_grad_recursively();
            }
            GradValOp::Add(a, b) => {
                a.borrow_mut().reset_grad_recursively();
                b.borrow_mut().reset_grad_recursively();
            }
            GradValOp::Sub(a, b) => {
                a.borrow_mut().reset_grad_recursively();
                b.borrow_mut().reset_grad_recursively();
            }
            GradValOp::Mul(a, b) => {
                a.borrow_mut().reset_grad_recursively();
                b.borrow_mut().reset_grad_recursively();
            }
            GradValOp::Div(a, b) => {
                a.borrow_mut().reset_grad_recursively();
                b.borrow_mut().reset_grad_recursively();
            }
        }
    }

    fn calc_grad_recursively(&mut self, grad: f32) {
        self._grad = Some(
            match self._grad {
                Some(g) => g,
                None => 0.,
            } + grad,
        );
        // Calc grad for children
        match &self._op {
            GradValOp::Noop => {}
            GradValOp::Neg(a) => {
                a.borrow_mut().calc_grad_recursively(-1.0);
            }
            GradValOp::Exp(a) => {
                let g = a.borrow()._val.exp();
                a.borrow_mut().calc_grad_recursively(g * grad);
            }
            GradValOp::Log(a) => {
                let g = 1. / a.borrow()._val;
                a.borrow_mut().calc_grad_recursively(g * grad);
            }
            GradValOp::Pow(a, b) => {
                let a_val = a.borrow()._val;
                let b_val = b.borrow()._val;
                let g = b_val * a_val.powf(b_val - 1.);
                a.borrow_mut().calc_grad_recursively(g * grad);
                let g = a_val.ln() * a_val.powf(b_val);
                b.borrow_mut().calc_grad_recursively(g * grad);
            }
            GradValOp::Add(a, b) => {
                a.borrow_mut().calc_grad_recursively(grad);
                b.borrow_mut().calc_grad_recursively(grad);
            }
            GradValOp::Sub(a, b) => {
                a.borrow_mut().calc_grad_recursively(grad);
                b.borrow_mut().calc_grad_recursively(-grad);
            }
            GradValOp::Mul(a, b) => {
                a.borrow_mut().calc_grad_recursively(grad * b.borrow()._val);
                b.borrow_mut().calc_grad_recursively(grad * a.borrow()._val);
            }
            GradValOp::Div(a, b) => {
                let a_val = a.borrow()._val;
                let b_val = b.borrow()._val;
                a.borrow_mut().calc_grad_recursively(grad / b_val);
                let g = -a_val / (b_val.powi(2));
                b.borrow_mut().calc_grad_recursively(g * grad);
            }
        }
    }
}

impl PartialEq for Gv {
    fn eq(&self, other: &Self) -> bool {
        self._val == other._val
    }
}

impl PartialOrd for Gv {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self._val.partial_cmp(&other._val)
    }
}

#[derive(Clone, Debug)]
pub struct GradVal {
    _gv: Rc<RefCell<Gv>>,
}

// Constructors
impl GradVal {
    pub fn from_value(v: f32) -> Self {
        GradVal {
            _gv: Rc::new(RefCell::new(Gv::from_value(v))),
        }
    }

    fn from_op(v: f32, op: GradValOp) -> Self {
        GradVal {
            _gv: Rc::new(RefCell::new(Gv::from_op(v, op))),
        }
    }
}

// Additional operators
impl GradVal {
    pub fn exp(&mut self) -> &Self {
        let old_self = self.clone();
        self._gv = Rc::new(RefCell::new(Gv::from_op(
            old_self._gv.borrow()._val.exp(),
            GradValOp::Exp(old_self._gv.clone()),
        )));
        self
    }

    pub fn log(&mut self) -> &Self {
        let old_self = self.clone();
        self._gv = Rc::new(RefCell::new(Gv::from_op(
            old_self._gv.borrow()._val.ln(),
            GradValOp::Log(old_self._gv.clone()),
        )));
        self
    }

    pub fn pow(&self, other: &Self) -> Self {
        GradVal {
            _gv: Rc::new(RefCell::new(Gv::from_op(
                RefCell::borrow(&self._gv)
                    ._val
                    .powf(RefCell::borrow(&other._gv)._val),
                GradValOp::Pow(self._gv.clone(), other._gv.clone()),
            ))),
        }
    }
}

// Backward propagation
impl GradVal {
    pub fn backward(&mut self) {
        RefCell::borrow_mut(&self._gv).reset_grad_recursively();
        RefCell::borrow_mut(&self._gv).calc_grad_recursively(1.);
    }

    pub fn grad(&self) -> Option<f32> {
        self._gv.borrow()._grad
    }
}

impl PartialEq for GradVal {
    fn eq(&self, other: &Self) -> bool {
        RefCell::borrow(&self._gv)._val == RefCell::borrow(&other._gv)._val
    }
}

impl PartialOrd for GradVal {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        RefCell::borrow(&self._gv)
            ._val
            .partial_cmp(&RefCell::borrow(&other._gv)._val)
    }
}

impl Neg for &GradVal {
    type Output = GradVal;

    fn neg(self) -> Self::Output {
        GradVal::from_op(
            -RefCell::borrow(&self._gv)._val,
            GradValOp::Neg(self._gv.clone()),
        )
    }
}

impl Add for &GradVal {
    type Output = GradVal;

    fn add(self, other: Self) -> Self::Output {
        GradVal {
            _gv: Rc::new(RefCell::new(Gv::from_op(
                RefCell::borrow(&self._gv)._val + RefCell::borrow(&other._gv)._val,
                GradValOp::Add(self._gv.clone(), other._gv.clone()),
            ))),
        }
    }
}

impl Sub for &GradVal {
    type Output = GradVal;

    fn sub(self, other: Self) -> Self::Output {
        GradVal {
            _gv: Rc::new(RefCell::new(Gv::from_op(
                RefCell::borrow(&self._gv)._val - RefCell::borrow(&other._gv)._val,
                GradValOp::Sub(self._gv.clone(), other._gv.clone()),
            ))),
        }
    }
}

impl Mul for &GradVal {
    type Output = GradVal;

    fn mul(self, other: Self) -> Self::Output {
        GradVal {
            _gv: Rc::new(RefCell::new(Gv::from_op(
                RefCell::borrow(&self._gv)._val * RefCell::borrow(&other._gv)._val,
                GradValOp::Mul(self._gv.clone(), other._gv.clone()),
            ))),
        }
    }
}

impl Div for &GradVal {
    type Output = GradVal;

    fn div(self, other: Self) -> Self::Output {
        let divider = RefCell::borrow(&other._gv)._val;
        if divider == 0. {
            panic!("Division by Zero :(");
        }
        GradVal {
            _gv: Rc::new(RefCell::new(Gv::from_op(
                RefCell::borrow(&self._gv)._val / divider,
                GradValOp::Div(self._gv.clone(), other._gv.clone()),
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neg() {
        let x = &GradVal::from_value(1.);
        let y = -x;
        assert_eq!(y, GradVal::from_value(-1.));
        assert_eq!(y._gv.borrow()._op, GradValOp::Neg(x._gv.clone()));
    }

    #[test]
    fn log() {
        let mut x = GradVal::from_value(1.);
        let y = x.clone();
        x.log();
        assert_eq!(x, GradVal::from_value(0.));
        assert_eq!(x._gv.borrow()._op, GradValOp::Log(y._gv));
    }

    #[test]
    fn exp() {
        let mut x = GradVal::from_value(0.);
        let y = x.clone();
        x.exp();
        assert_eq!(x, GradVal::from_value(1.));
        assert_eq!(x._gv.borrow()._op, GradValOp::Exp(y._gv));
    }

    #[test]
    fn pow() {
        let x = GradVal::from_value(2.0);
        let y = GradVal::from_value(3.);
        let z = x.pow(&y);
        assert_eq!(z, GradVal::from_value(8.));
        assert_eq!(z._gv.borrow()._op, GradValOp::Pow(x._gv, y._gv));
    }

    #[test]
    fn add() {
        let x = GradVal::from_value(1.0);
        let y = GradVal::from_value(2.0);
        let z = &x + &y;
        assert_eq!(z, GradVal::from_value(3.0));
        assert_eq!(z._gv.borrow()._op, GradValOp::Add(x._gv, y._gv));
    }

    #[test]
    fn sub() {
        let x = GradVal::from_value(1.0);
        let y = GradVal::from_value(2.0);
        let z = &x - &y;
        assert_eq!(z, GradVal::from_value(-1.0));
        assert_eq!(z._gv.borrow()._op, GradValOp::Sub(x._gv, y._gv));
    }

    #[test]
    fn mul() {
        let x = GradVal::from_value(2.0);
        let y = GradVal::from_value(3.0);
        let z = &x * &y;
        assert_eq!(z, GradVal::from_value(6.0));
        assert_eq!(z._gv.borrow()._op, GradValOp::Mul(x._gv, y._gv));
    }

    #[test]
    fn div() {
        let x = GradVal::from_value(4.0);
        let y = GradVal::from_value(2.0);
        let z = &x / &y;
        assert_eq!(z, GradVal::from_value(2.0));
        assert_eq!(z._gv.borrow()._op, GradValOp::Div(x._gv, y._gv));
    }

    #[test]
    #[should_panic]
    fn div_zero() {
        let x = &GradVal::from_value(1.);
        let y = &GradVal::from_value(0.);
        let _ = x / y;
    }

    #[test]
    fn long_expression() {
        let a = &GradVal::from_value(1.0);
        let b = &GradVal::from_value(2.0);
        let c = &GradVal::from_value(3.0);
        let p = &GradVal::from_value(3.0);
        let z = (-&(&(a + c) - b)).pow(p);
        assert_eq!(z, GradVal::from_value(-8.0));
    }

    #[test]
    fn grad_neg() {
        let x = GradVal::from_value(2.0);
        let mut z = -&x;
        z.backward();
        assert_eq!(z.grad(), Some(1.));
        assert_eq!(x.grad(), Some(-1.));
    }

    #[test]
    fn grad_exp() {
        let x = GradVal::from_value(2.0);
        let mut z = x.clone();
        z.exp();
        z.backward();
        assert_eq!(z.grad(), Some(1.));
        assert_eq!(x.grad(), Some(2_f32.exp()));
    }

    #[test]
    fn grad_log() {
        let x = GradVal::from_value(2.0);
        let mut z = x.clone();
        z.log();
        z.backward();
        assert_eq!(z.grad(), Some(1.));
        assert_eq!(x.grad(), Some(0.5));
    }

    #[test]
    fn grad_pow() {
        let x = GradVal::from_value(2.0);
        let y = GradVal::from_value(3.0);
        let mut z = x.pow(&y);
        z.backward();
        assert_eq!(z.grad(), Some(1.));
        assert_eq!(x.grad(), Some(12.));
        assert_eq!(y.grad(), Some(2_f32.ln() * 8.));
    }

    #[test]
    fn grad_add() {
        let x = GradVal::from_value(2.0);
        let y = GradVal::from_value(3.0);
        let mut z = &x + &y;
        z.backward();
        assert_eq!(z.grad(), Some(1.));
        assert_eq!(x.grad(), Some(1.));
        assert_eq!(y.grad(), Some(1.));
    }

    #[test]
    fn grad_sub() {
        let x = GradVal::from_value(2.0);
        let y = GradVal::from_value(3.0);
        let mut z = &x - &y;
        z.backward();
        assert_eq!(z.grad(), Some(1.));
        assert_eq!(x.grad(), Some(1.));
        assert_eq!(y.grad(), Some(-1.));
    }

    #[test]
    fn grad_mul() {
        let x = GradVal::from_value(2.0);
        let y = GradVal::from_value(3.0);
        let mut z = &x * &y;
        z.backward();
        assert_eq!(z.grad(), Some(1.));
        assert_eq!(x.grad(), Some(3.));
        assert_eq!(y.grad(), Some(2.));
    }

    #[test]
    fn grad_div() {
        let x = GradVal::from_value(2.0);
        let y = GradVal::from_value(4.0);
        let mut z = &x / &y;
        z.backward();
        assert_eq!(z.grad(), Some(1.));
        assert_eq!(x.grad(), Some(0.25));
        assert_eq!(y.grad(), Some(-0.125));
    }
}
