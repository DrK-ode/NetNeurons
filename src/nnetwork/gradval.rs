use std::{
    cell::RefCell,
    ops::{Add, Div, Mul, Neg, Sub},
    rc::Rc,
};

type Ancestor = Rc<RefCell<Gv>>;

#[derive(Clone, Debug, PartialEq)]
enum GradValOp {
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
        gv._val = gv._val.exp();
        gv._op = Some(GradValOp::Exp(old_self._gv));
        self
    }

    pub fn log(&mut self) -> &Self {
        let old_self = self.clone();
        let mut gv = self._gv.borrow_mut();
        gv._val = gv._val.ln();
        gv._op = Some(GradValOp::Log(old_self._gv));
        self
    }

    pub fn pow(&self, other: &Self) -> Self {
        GradVal {
            _gv: Rc::new(RefCell::new(Gv::from_op(
                RefCell::borrow(&self._gv)._val.powf(RefCell::borrow(&other._gv)._val),
                GradValOp::Pow(self._gv.clone(), other._gv.clone()),
            ))),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neg() {
        let x = &GradVal::from_value(1.);
        let y = -x;
        assert_eq!(y, GradVal::from_value(-1.));
        assert_eq!(y._gv.borrow()._op, Some(GradValOp::Neg(x._gv.clone())));
    }

    #[test]
    fn log() {
        let mut x = GradVal::from_value(1.);
        let y = x.clone();
        x.log();
        assert_eq!(x, GradVal::from_value(0.));
        assert_eq!(x._gv.borrow()._op, Some(GradValOp::Log(y._gv)));
    }

    #[test]
    fn exp() {
        let mut x = GradVal::from_value(0.);
        let y = x.clone();
        x.exp();
        assert_eq!(x, GradVal::from_value(1.));
        assert_eq!(x._gv.borrow()._op, Some(GradValOp::Exp(y._gv)));}

    #[test]
    fn pow() {
        let x = GradVal::from_value(2.0);
        let y = GradVal::from_value(3.);
        let z = x.pow(&y);
        assert_eq!(z, GradVal::from_value(8.));
        assert_eq!(z._gv.borrow()._op, Some(GradValOp::Pow(x._gv,y._gv)));}

    #[test]
    fn add() {
        let x = GradVal::from_value(1.0);
        let y = GradVal::from_value(2.0);
        let z = &x + &y;
        assert_eq!(z, GradVal::from_value(3.0));
        assert_eq!(
            z._gv.borrow()._op,
            Some(GradValOp::Add(x._gv, y._gv))
        );
    }

    #[test]
    fn sub() {
        let x = GradVal::from_value(1.0);
        let y = GradVal::from_value(2.0);
        let z = &x - &y;
        assert_eq!(z, GradVal::from_value(-1.0));
        assert_eq!(
            z._gv.borrow()._op,
            Some(GradValOp::Sub(x._gv, y._gv))
        );
    }

    #[test]
    fn mul() {
        let x = GradVal::from_value(2.0);
        let y = GradVal::from_value(3.0);
        let z = &x * &y;
        assert_eq!(z, GradVal::from_value(6.0));
        assert_eq!(
            z._gv.borrow()._op,
            Some(GradValOp::Mul(x._gv, y._gv))
        );
    }

    #[test]
    fn div() {
        let x = GradVal::from_value(4.0);
        let y = GradVal::from_value(2.0);
        let z = &x / &y;
        assert_eq!(z, GradVal::from_value(2.0));
        assert_eq!(
            z._gv.borrow()._op,
            Some(GradValOp::Div(x._gv, y._gv))
        );
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
        let z = (-&(&(a+c)-b)).pow(p);
        assert_eq!(z, GradVal::from_value(-8.0));
    }
}
