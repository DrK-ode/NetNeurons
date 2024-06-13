use std::{
    cell::RefCell,
    ops::{Add, Div, Mul, Neg, Sub},
    rc::Rc,
};

type Ancestor = Rc<RefCell<Gv>>;

#[derive(Clone)]
enum GradValOp {
    Neg(Ancestor),
    Exp(Ancestor),
    Pow(Ancestor, f32),
    Add(Ancestor, Ancestor),
    Mul(Ancestor, Ancestor),
}

#[derive(Default)]
struct Gv {
    _val: f32,
    _grad: Option<f32>,
    _op: Option<GradValOp>,
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
            _op: Some(op),
            ..Self::default()
        }
    }
}

#[derive(Clone)]
pub struct GradVal {
    _gv: Rc<RefCell<Gv>>,
}

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

    pub fn exp(&mut self) -> &Self {
        let old_self = self.clone();
        let mut gv = self._gv.borrow_mut();
        gv._val = RefCell::borrow(&self._gv)._val.exp();
        gv._op = Some(GradValOp::Exp(old_self._gv));
        self
    }

    pub fn pow(&mut self, p: f32) -> &Self {
        let old_self = self.clone();
        let mut gv = self._gv.borrow_mut();
        gv._val = RefCell::borrow(&self._gv)._val.powf(p);
        gv._op = Some(GradValOp::Pow(old_self._gv, p));
        self
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
        self + &(-other)
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
                GradValOp::Mul(self._gv.clone(), other._gv.clone()),
            ))),
        }
    }
}
