use super::*;

#[test]
fn neg() {
    let x = &GradVal::from(1.);
    let y = -x;
    assert_eq!(y, GradVal::from(-1.));
    assert_eq!(y._gv.borrow()._op, GradValOp::Neg(x._gv.clone()));
}

#[test]
fn log() {
    let x = GradVal::from(1.);
    let z = x.log();
    assert_eq!(z, 0_f32.into());
    assert_eq!(z._gv.borrow()._op, GradValOp::Log(x._gv));
}

#[test]
fn exp() {
    let x = GradVal::from(0.);
    let z = x.exp();
    assert_eq!(z, 1_f32.into());
    assert_eq!(z._gv.borrow()._op, GradValOp::Exp(x._gv));
}

#[test]
fn pow() {
    let x = GradVal::from(2.0);
    let y = GradVal::from(3.);
    let z = x.pow(&y);
    assert_eq!(z, 8_f32.into());
    assert_eq!(z._gv.borrow()._op, GradValOp::Pow(x._gv, y._gv));
}

#[test]
fn add() {
    let x = GradVal::from(1.0);
    let y = GradVal::from(2.0);
    let z = &x + &y;
    assert_eq!(z, 3_f32.into());
    assert_eq!(z._gv.borrow()._op, GradValOp::Add(x._gv, y._gv));
}

#[test]
fn sub() {
    let x = GradVal::from(1.0);
    let y = GradVal::from(2.0);
    let z = &x - &y;
    assert_eq!(z, (-1_f32).into());
    assert_eq!(z._gv.borrow()._op, GradValOp::Sub(x._gv, y._gv));
}

#[test]
fn mul() {
    let x = GradVal::from(2.0);
    let y = GradVal::from(3.0);
    let z = &x * &y;
    assert_eq!(z, 6_f32.into());
    assert_eq!(z._gv.borrow()._op, GradValOp::Mul(x._gv, y._gv));
}

#[test]
fn div() {
    let x = GradVal::from(4.0);
    let y = GradVal::from(2.0);
    let z = &x / &y;
    assert_eq!(z, 2_f32.into());
    assert_eq!(z._gv.borrow()._op, GradValOp::Div(x._gv, y._gv));
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
    let c = &GradVal::from(2.0);
    let d = &(&(a + b) - c);
    let z: GradVal = (-d).pow(&3_f32.into()).log().exp();
    assert_eq!(z, 8_f32.into());
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
