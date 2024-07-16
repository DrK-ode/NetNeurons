use std::{
    cell::RefCell,
    fmt::Display,
    iter::Sum,
    ops::{Add, Div, Mul, Neg, Sub},
    rc::Rc,
};

type Ancestor = Rc<RefCell<Gv>>;

#[derive(Clone, Debug, Default, PartialEq)]
enum GradValOp {
    #[default]
    Noop,
    Exp(Ancestor),
    Log(Ancestor),
    Pow(Ancestor, Ancestor),
    Add(Ancestor, Ancestor),
    Mul(Ancestor, Ancestor),
    Sum(Vec<Ancestor>),
}
impl GradValOp {
    fn op_symb(&self) -> &str {
        match self {
            GradValOp::Noop => "NOOP",
            GradValOp::Exp(_) => "exp",
            GradValOp::Log(_) => "log",
            GradValOp::Pow(_, _) => "^",
            GradValOp::Add(_, _) => "+",
            GradValOp::Mul(_, _) => "*",
            GradValOp::Sum(_) => "sum",
        }
    }
}

impl Display for GradValOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GradValOp::Noop => write!(f, "{}", self.op_symb()),
            GradValOp::Exp(a) | GradValOp::Log(a) => {
                write!(f, "{}({:e})", self.op_symb(), a.borrow()._val)
            }
            GradValOp::Pow(a, b) | GradValOp::Add(a, b) | GradValOp::Mul(a, b) => write!(
                f,
                "{:e} {} {:e}",
                a.borrow()._val,
                self.op_symb(),
                b.borrow()._val
            ),
            GradValOp::Sum(vec) => {
                write!(
                    f,
                    "{}({})",
                    self.op_symb(),
                    vec.iter()
                        .map(|gv| gv.borrow()._val.to_string() + ", ")
                        .collect::<String>()
                )
            }
        }
    }
}

#[derive(Debug, Default)]
struct Gv {
    _val: f32,
    _grad: Option<f32>, // Partial derivative of root value having called backward() wrt. this value
    _op: GradValOp,     // Operation which the value originated from
}

impl PartialEq for Gv {
    fn eq(&self, other: &Self) -> bool {
        self as *const Self == other as *const Self
    }
}

impl Eq for Gv {}

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
impl From<f32> for Gv {
    fn from(value: f32) -> Self {
        Gv {
            _val: value,
            ..Self::default()
        }
    }
}

// Back propagation
impl Gv {
    fn calc_grad(&self) {
        fn add_grad(gv: &Rc<RefCell<Gv>>, new_grad: f32) {
            let old_grad = gv.borrow()._grad.unwrap_or(0.);
            gv.borrow_mut()._grad = Some(old_grad + new_grad);
        }
        // Calc grad for children
        let grad = self._grad.unwrap();
        match &self._op {
            GradValOp::Noop => {}
            GradValOp::Exp(a) => {
                let g = self._val;
                add_grad(a, g * grad);
            }
            GradValOp::Log(a) => {
                let g = 1. / a.borrow()._val;
                add_grad(a, g * grad);
            }
            GradValOp::Pow(a, b) => {
                let a_val = a.borrow()._val;
                let b_val = b.borrow()._val;
                let g = b_val * a_val.powf(b_val - 1.);
                add_grad(a, g * grad);
                let g = a_val.ln() * a_val.powf(b_val);
                add_grad(b, g * grad);
            }
            GradValOp::Add(a, b) => {
                add_grad(a, grad);
                add_grad(b, grad);
            }
            GradValOp::Mul(a, b) => {
                let g = b.borrow()._val;
                add_grad(a, g * grad);
                let g = a.borrow()._val;
                add_grad(b, g * grad);
            }
            GradValOp::Sum(vec) => {
                vec.iter().for_each(|gv| add_grad(gv, grad));
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct GradVal {
    _gv: Rc<RefCell<Gv>>,
}

impl Display for GradVal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({:e}", self.value())?;
        if let Some(g) = self.grad() {
            write!(f, ", ∇: {:e}", g)?;
        }
        write!(f, ")")
    }
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
impl From<&GradVal> for f32 {
    fn from(gv: &GradVal) -> Self {
        gv._gv.borrow()._val
    }
}

impl PartialEq for GradVal {
    fn eq(&self, other: &Self) -> bool {
        self.value() == other.value()
    }
}

impl PartialOrd for GradVal {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.value().partial_cmp(&other.value())
    }
}

impl Neg for &GradVal {
    type Output = GradVal;

    fn neg(self) -> Self::Output {
        self * &GradVal::from(-1.)
    }
}

impl Neg for GradVal {
    type Output = GradVal;

    fn neg(self) -> Self::Output {
        self * GradVal::from(-1.)
    }
}

impl Add for &GradVal {
    type Output = GradVal;

    fn add(self, other: Self) -> Self::Output {
        GradVal::from_op(
            self.value() + other.value(),
            GradValOp::Add(self._gv.clone(), other._gv.clone()),
        )
    }
}

impl Add for GradVal {
    type Output = GradVal;

    fn add(self, other: Self) -> Self::Output {
        GradVal::from_op(
            self.value() + other.value(),
            GradValOp::Add(self._gv.clone(), other._gv.clone()),
        )
    }
}

impl Sub for &GradVal {
    type Output = GradVal;

    fn sub(self, other: Self) -> Self::Output {
        self + &(-other)
    }
}

impl Sub for GradVal {
    type Output = GradVal;

    fn sub(self, other: Self) -> Self::Output {
        self + (-other)
    }
}

impl Mul for &GradVal {
    type Output = GradVal;

    fn mul(self, other: Self) -> Self::Output {
        GradVal::from_op(
            self.value() * other.value(),
            GradValOp::Mul(self._gv.clone(), other._gv.clone()),
        )
    }
}

impl Mul for GradVal {
    type Output = GradVal;

    fn mul(self, other: Self) -> Self::Output {
        GradVal::from_op(
            self.value() * other.value(),
            GradValOp::Mul(self._gv.clone(), other._gv.clone()),
        )
    }
}

impl Div for &GradVal {
    type Output = GradVal;

    fn div(self, other: Self) -> Self::Output {
        let divider = other.value();
        if divider == 0. {
            panic!("Division not defined for zero divider");
        }
        self * &other.powf(-1.)
    }
}

impl Div for GradVal {
    type Output = GradVal;

    fn div(self, other: Self) -> Self::Output {
        let divider = other.value();
        if divider == 0. {
            panic!("Division not defined for zero divider");
        }
        self * other.powf(-1.)
    }
}

impl Sum for GradVal {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut result = 0.;
        let vec: Vec<_> = iter
            .map(|value| {
                result += value.value();
                value._gv.clone()
            })
            .collect();
        GradVal::from_op(result, GradValOp::Sum(vec))
    }
}

// Additional operators
impl GradVal {
    pub fn exp(&self) -> Self {
        GradVal::from_op(self.value().exp(), GradValOp::Exp(self._gv.clone()))
    }

    pub fn log(&self) -> Self {
        let arg = f32::from(self);
        if arg <= 0. {
            panic!("Log not defined for arg <= 0, value provided was {arg}");
        }
        GradVal::from_op(arg.ln(), GradValOp::Log(self._gv.clone()))
    }

    pub fn pow(&self, other: &GradVal) -> GradVal {
        GradVal::from_op(
            self.value().powf(other.value()),
            GradValOp::Pow(self._gv.clone(), other._gv.clone()),
        )
    }

    pub fn powf(&self, other: f32) -> Self {
        let other = GradVal::from(other);
        return self.pow(&other);
    }

    pub fn sigmoid(&self) -> Self {
        &GradVal::from(1.) / &(&GradVal::from(1.) + &(-self).exp())
    }

    pub fn sum(vec: &Vec<GradVal>) -> GradVal {
        GradVal::from_op(
            vec.iter().fold(0., |acc, v| acc + v.value()),
            GradValOp::Sum(vec.iter().map(|value| value._gv.clone()).collect()),
        )
    }
}

// Backward propagation
impl GradVal {
    pub fn backward(&mut self) {
        fn collect_and_clear(
            gv: &Rc<RefCell<Gv>>,
            visited: &mut Vec<Rc<RefCell<Gv>>>,
            gvs: &mut Vec<Rc<RefCell<Gv>>>,
        ) {
            if !visited.contains(&gv) {
                // Clear grad before new calc
                gv.borrow_mut()._grad = None;
                visited.push(gv.clone());
                match &gv.borrow()._op {
                    GradValOp::Noop => {
                        return ();
                    }
                    GradValOp::Exp(a) | GradValOp::Log(a) => collect_and_clear(a, visited, gvs),
                    GradValOp::Pow(a, b) | GradValOp::Add(a, b) | GradValOp::Mul(a, b) => {
                        collect_and_clear(&a, visited, gvs);
                        collect_and_clear(&b, visited, gvs);
                    }
                    GradValOp::Sum(vec) => {
                        vec.iter()
                            .for_each(|gv| collect_and_clear(gv, visited, gvs));
                    }
                }
                gvs.push(gv.clone());
            }
        }
        let mut visited: Vec<Rc<RefCell<Gv>>> = Vec::new();
        let mut gvs: Vec<Rc<RefCell<Gv>>> = Vec::new();
        collect_and_clear(&self._gv, &mut visited, &mut gvs);

        // Set gradient for root value to 1
        gvs.last().unwrap().borrow_mut()._grad = Some(1.);
        for gv in gvs.iter().rev() {
            gv.borrow().calc_grad();
        }
    }
}

// Access functions
impl GradVal {
    pub fn value(&self) -> f32 {
        f32::from(self)
    }

    pub fn set_value(&mut self, v: f32) {
        self._gv.borrow_mut()._val = v;
    }

    pub fn grad(&self) -> Option<f32> {
        self._gv.borrow()._grad
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn neg() {
        let x = &GradVal::from(1.);
        let y = -x;
        assert_eq!(y.value(), -1.);
    }

    #[test]
    fn log() {
        let x = GradVal::from(1.);
        let z = x.log();
        assert_eq!(z.value(), 0.);
        assert_eq!(z._gv.borrow()._op, GradValOp::Log(x._gv));
    }

    #[test]
    fn exp() {
        let x = GradVal::from(0.);
        let z = x.exp();
        assert_eq!(z.value(), 1.);
        assert_eq!(z._gv.borrow()._op, GradValOp::Exp(x._gv));
    }

    #[test]
    fn pow() {
        let x = GradVal::from(2.0);
        let y = GradVal::from(3.);
        let z = x.pow(&y);
        assert_eq!(z.value(), 8.);
        assert_eq!(z._gv.borrow()._op, GradValOp::Pow(x._gv, y._gv));
    }

    #[test]
    fn add() {
        let x = GradVal::from(1.0);
        let y = GradVal::from(2.0);
        let z = &x + &y;
        assert_eq!(z.value(), 3.);
        assert_eq!(z._gv.borrow()._op, GradValOp::Add(x._gv, y._gv));
    }

    #[test]
    fn sub() {
        let x = GradVal::from(1.0);
        let y = GradVal::from(2.0);
        let z = x - y;
        assert_eq!(z.value(), -1.);
    }

    #[test]
    fn mul() {
        let x = GradVal::from(2.0);
        let y = GradVal::from(3.0);
        let z = &x * &y;
        assert_eq!(z.value(), 6.);
        assert_eq!(z._gv.borrow()._op, GradValOp::Mul(x._gv, y._gv));
    }

    #[test]
    fn div() {
        let x = GradVal::from(4.0);
        let y = GradVal::from(2.0);
        let z = x / y;
        assert_eq!(z.value(), 2.);
    }

    #[test]
    #[should_panic]
    fn div_zero() {
        let x = &GradVal::from(1.);
        let y = &GradVal::from(0.);
        let _ = x / y;
    }

    #[test]
    fn sum() {
        let x = GradVal::from(1.0) + GradVal::from(2.0) + GradVal::from(3.0);
        let y = GradVal::sum(&vec![
            GradVal::from(1.0),
            GradVal::from(2.0),
            GradVal::from(3.0),
        ]);
        assert_eq!(x.value(), y.value());
        assert_eq!(x.grad(), y.grad());
    }

    #[test]
    fn long_expression() {
        let a = &GradVal::from(1.0);
        let b = &GradVal::from(-1.0);
        let c = &GradVal::from(1.0);
        let d = &(&(a + b) - &(c * &2_f32.into()));
        let z: GradVal = (-d).powf(3.).log().exp();
        assert_eq!(z.value(), 8.);
    }

    #[test]
    fn grad_neg() {
        let x = GradVal::from(2.0);
        let mut z = -&x;
        z.backward();
        assert_eq!(z.grad(), Some(1.));
        assert_eq!(x.grad(), Some(-1.));
    }

    #[test]
    fn grad_exp() {
        let x = GradVal::from(2.0);
        let mut z = x.exp();
        z.backward();
        assert_eq!(z.grad(), Some(1.));
        assert_eq!(x.grad(), Some(2_f32.exp()));
    }

    #[test]
    fn grad_log() {
        let x = GradVal::from(2.0);
        let mut z = x.log();
        z.backward();
        assert_eq!(z.grad(), Some(1.));
        assert_eq!(x.grad(), Some(0.5));
    }

    #[test]
    fn grad_pow() {
        let x = GradVal::from(2.0);
        let y = GradVal::from(3.0);
        let mut z = x.pow(&y);
        z.backward();
        assert_eq!(z.grad(), Some(1.));
        assert_eq!(x.grad(), Some(12.));
        assert_eq!(y.grad(), Some(2_f32.ln() * 8.));
    }

    #[test]
    fn grad_add() {
        let x = GradVal::from(2.0);
        let y = GradVal::from(3.0);
        let mut z = &x + &y;
        z.backward();
        assert_eq!(z.grad(), Some(1.));
        assert_eq!(x.grad(), Some(1.));
        assert_eq!(y.grad(), Some(1.));
    }

    #[test]
    fn grad_sub() {
        let x = GradVal::from(2.0);
        let y = GradVal::from(3.0);
        let mut z = &x - &y;
        z.backward();
        assert_eq!(z.grad(), Some(1.));
        assert_eq!(x.grad(), Some(1.));
        assert_eq!(y.grad(), Some(-1.));
    }

    #[test]
    fn grad_mul() {
        let x = GradVal::from(2.0);
        let y = GradVal::from(3.0);
        let mut z = &x * &y;
        z.backward();
        assert_eq!(z.grad(), Some(1.));
        assert_eq!(x.grad(), Some(3.));
        assert_eq!(y.grad(), Some(2.));
    }

    #[test]
    fn grad_div() {
        let x = GradVal::from(2.0);
        let y = GradVal::from(4.0);
        let mut z = &x / &y;
        z.backward();
        assert_eq!(z.grad(), Some(1.));
        assert_eq!(x.grad(), Some(0.25));
        assert_eq!(y.grad(), Some(-0.125));
    }

    #[test]
    fn long_grad() {
        let a = &GradVal::from(1.0);
        let b = &GradVal::from(-1.0);
        let c = &GradVal::from(1.0);
        let d = &(&(a + b) - &(c * &2_f32.into())); // = -2
        let e = &(-d).powf(3.); // = 8
        let f = &e.log(); // = log(8)
        let mut g = f.exp(); // = 8

        g.backward();

        assert_eq!(g.grad(), Some(1.0));
        assert_eq!(f.grad(), Some(8.));
        assert_eq!(e.grad(), Some(1.0));
        assert_eq!(d.grad(), Some(-12.0));
        assert_eq!(a.grad(), Some(-12.));
        assert_eq!(b.grad(), Some(-12.));
        assert_eq!(c.grad(), Some(24.));
    }

    #[test]
    fn same_value_many_times() {
        let a = &(&GradVal::from(3.0) - &GradVal::from(1.0));
        let b = &GradVal::from(3.0);
        let c = &(&(a * a) + b);
        let d = &(&(c / a) + a);
        let mut z = d * d;
        z.backward();
        assert_eq!(b.grad().unwrap(), 5.5);
        assert_eq!(a.grad().unwrap(), 13.75);
    }

    #[test]
    fn equality() {
        let a = &GradVal::from(1.0);
        let b = &GradVal::from(1.0);
        let c = a.clone();
        assert_ne!(a._gv, b._gv);
        assert_eq!(a._gv, c._gv);
    }
}
