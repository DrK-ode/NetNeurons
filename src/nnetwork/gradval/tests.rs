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
    let z = &x - &y;
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
    let z = &x / &y;
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
fn long_expression() {
    let a = &GradVal::from(1.0);
    let b = &GradVal::from(-1.0);
    let c = &GradVal::from(1.0);
    let d = &(&(a + b) - &(c*&2_f32.into()));
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
    let d = &(&(a + b) - &(c*&2_f32.into()) ); // = -2
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
    let c = &(&(a*a) + b);
    let d = &(&(c / a) + a);
    let mut z = d*d;
    z.backward();
    assert_eq!(b.grad().unwrap(), 5.5);
    assert_eq!(a.grad().unwrap(), 13.75);
}

#[test]
fn equality() {
    let a = &GradVal::from(1.0);
    let b = &GradVal::from(1.0);
    let c = a.clone();
    assert_ne!(a._gv,b._gv);
    assert_eq!(a._gv,c._gv);
}
