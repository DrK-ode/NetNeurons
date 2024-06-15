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

// Constructors
impl Gv {
    fn from_op(v: f32, op: GradValOp) -> Self {
        Gv {
            _val: v,
            _op: op,
            ..Self::default()
        }
    }
}
impl From<f32> for Gv{
    fn from(value: f32) -> Self {
        Gv {
            _val: value,
            ..Self::default()
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

// Back propagation
impl Gv{
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

#[derive(Clone, Debug)]
pub struct GradVal {
    _gv: Rc<RefCell<Gv>>,
}

// Constructors
impl GradVal {
    fn from_op(v: f32, op: GradValOp) -> Self {
        GradVal {
            _gv: Rc::new(RefCell::new(Gv::from_op(v, op))),
        }
    }
}
impl From<f32> for GradVal {
    fn from(value: f32) -> Self {
        GradVal {
            _gv: Rc::new(RefCell::new(value.into())),
        }
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

// Additional operators
impl GradVal {
    pub fn exp(&self) -> Self {
        GradVal::from_op(self._gv.borrow()._val.exp(), GradValOp::Exp(self._gv.clone()))
    }

    pub fn log(&self) -> Self {
        GradVal::from_op(self._gv.borrow()._val.ln(), GradValOp::Log(self._gv.clone()))
    }

    pub fn pow(&self, other: &Self) -> Self {
        GradVal::from_op(self._gv.borrow()._val.powf(other._gv.borrow()._val),
                GradValOp::Pow(self._gv.clone(), other._gv.clone()) )
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

#[cfg(test)]
mod tests;