use std::{
    cell::RefCell, fmt::Display, iter::Sum, ops::{Add, Div, Mul, Neg, Sub}, rc::Rc
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
    fn reset_grad_recursively(&mut self) {
        self._grad = None;
        match &self._op {
            GradValOp::Noop => {}
            GradValOp::Exp(a) | GradValOp::Log(a) => a.borrow_mut().reset_grad_recursively(),
            GradValOp::Pow(a, b) | GradValOp::Add(a, b) | GradValOp::Mul(a, b) => {
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
            GradValOp::Exp(a) => {
                let g = self._val;
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
            GradValOp::Mul(a, b) => {
                let g = b.borrow()._val;
                a.borrow_mut().calc_grad_recursively(g * grad);
                let g = a.borrow()._val;
                b.borrow_mut().calc_grad_recursively(g * grad);
            }
        }
    }

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
        }
    }
}

#[derive(Clone, Debug)]
pub struct GradVal {
    _gv: Rc<RefCell<Gv>>,
}

impl Display for GradVal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[GVal: ")?;
        if self._gv.borrow()._op != GradValOp::Noop {
            write!(f, " {} = ", self._gv.borrow()._op)?;
        }
        write!(f, "{:e}", f32::from(self))?;
        if self.grad().is_some() {
            write!(f, ", ∇: {:e}", self.grad().unwrap())?;
        }
        write!(f, "]")
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

/*impl PartialEq for GradVal {
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
}*/

impl Neg for &GradVal {
    type Output = GradVal;

    fn neg(self) -> Self::Output {
        self * &GradVal::from(-1.)
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

impl Sub for &GradVal {
    type Output = GradVal;

    fn sub(self, other: Self) -> Self::Output {
        self + &(-other)
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

impl Sum for GradVal{
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(GradVal::from(0.), |a,b| &a + &b )
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
            panic!("Log not defined for arg <= 0");
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
}

// Backward propagation
impl GradVal {
    // Recursive method is slower and might be removed soon
    #[allow(dead_code)]
    fn backward_recursively(&mut self) {
        RefCell::borrow_mut(&self._gv).reset_grad_recursively();
        RefCell::borrow_mut(&self._gv).calc_grad_recursively(1.);
    }

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
    pub fn grad(&self) -> Option<f32> {
        self._gv.borrow()._grad
    }
}

#[cfg(test)]
mod tests;
