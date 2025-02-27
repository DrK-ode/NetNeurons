use std::{
    iter::Sum,
    ops::{Add, Div, Mul, Neg, Sub},
};

use rand::Rng;

use super::{CalcNode, NodeType};

impl Sum for CalcNode {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|sum, node| sum + node).unwrap()
    }
}

impl CalcNode {
    /// Returns the sum of all values in the [CalcNode], i.e., the result will be a scalar.
    pub fn sum(&self) -> CalcNode {
        let result = Self::new_scalar(self.borrow()._vals.iter().sum());
        result.borrow_mut()._parent_nodes = vec![self.clone()];
        result.borrow_mut()._back_propagation = Some(Box::new(|child| {
            let child_grad = child.gradient_indexed(0);
            child.copy_parents()[0]
                .borrow_mut()
                ._grad
                .iter_mut()
                .for_each(|g| *g += child_grad);
        }));
        result
    }

    /// Normalises the [CalcNode] so that its sum becomes unity.
    pub fn normalized(&self) -> CalcNode {
        self / self.sum()
    }
    
    /// Collapses the [CalcNode] to a one-hot vector. Only makes sense if you can interpret the values as a probabilities that sum to unity.
    pub fn collapse(&self) -> CalcNode {
        let mut vec = vec![0.; self.len()];
        let mut rnd = rand::thread_rng().gen_range(0. ..self.borrow()._vals.iter().sum());
        for (i, &v) in self.borrow()._vals.iter().enumerate() {
            rnd -= v;
            if rnd <= 0. || i + 1 == self.len() {
                // Safe-guard against float precision errors
                vec[i] = 1.;
                break;
            }
        }
        CalcNode::new_from_shape(self.shape(), vec)
    }
}

/// Addition is implemented for all combinations of [CalcNode] and &[CalcNode].
impl Add<&CalcNode> for CalcNode {
    type Output = CalcNode;

    fn add(self, rhs: &CalcNode) -> Self::Output {
        &self + rhs
    }
}
impl Add for CalcNode {
    type Output = CalcNode;

    fn add(self, rhs: CalcNode) -> Self::Output {
        &self + &rhs
    }
}
impl Add<CalcNode> for &CalcNode {
    type Output = CalcNode;

    fn add(self, rhs: CalcNode) -> Self::Output {
        self + &rhs
    }
}
impl Add for &CalcNode {
    type Output = CalcNode;

    fn add(self, rhs: Self) -> Self::Output {
        // If self is a scalar make it the RHS
        let a = if self.len() == 1 { rhs } else { self };
        let b = if self.len() == 1 { self } else { rhs };
        // Adding scalar
        let result: Vec<_> = if b.len() == 1 {
            let b = b.value_indexed(0);
            a.borrow()._vals.iter().map(|a| a + b).collect()
        }
        // Compatible shapes
        else if a.len() == b.len() {
            a.borrow()
                ._vals
                .iter()
                .zip(b.borrow()._vals.iter())
                .map(|(a, b)| a + b)
                .collect()
        } else {
            panic!("Invalid operands for addition {a} and {b}.");
        };
        let result = CalcNode::new_from_shape(a.shape(), result);
        result.borrow_mut()._parent_nodes = vec![a.clone(), b.clone()];
        result.borrow_mut()._back_propagation = Some(Box::new(|child| {
            let parents = &child.borrow()._parent_nodes;
            for p in parents {
                p.borrow_mut()
                    ._grad
                    .iter_mut()
                    .zip(child.borrow()._grad.iter())
                    .for_each(|(p, c)| *p += c);
            }
        }));
        result
    }
}

/// Multiplication is implemented for all combinations of [CalcNode] and &[CalcNode].
/// 
/// Two kinds of multiplications are recognised:
/// 1. multiplication by a scalar, and
/// 2. the dot product.
/// 
/// Element-wise multiplication is provided by [CalcNode::element_wise_mul].
/// 
/// Will panic if the dimensions of LHS and RHS does not match.
impl Mul<&CalcNode> for CalcNode {
    type Output = CalcNode;

    fn mul(self, rhs: &CalcNode) -> Self::Output {
        &self * rhs
    }
}
impl Mul<CalcNode> for CalcNode {
    type Output = CalcNode;

    fn mul(self, rhs: CalcNode) -> Self::Output {
        &self * &rhs
    }
}
impl Mul<CalcNode> for &CalcNode {
    type Output = CalcNode;

    fn mul(self, rhs: CalcNode) -> Self::Output {
        self * &rhs
    }
}
impl Mul for &CalcNode {
    type Output = CalcNode;

    fn mul(self, other: Self) -> Self::Output {
        // If self is a scalar make it the RHS
        let a = if self.len() == 1 { other } else { self };
        let b = if self.len() == 1 { self } else { other };
        // Multiplying with scalar
        if b.len() == 1 {
            let scalar = b.value_indexed(0);
            let result = a.borrow()._vals.iter().map(|a| a * scalar).collect();
            let result = CalcNode::new_from_shape(a.shape(), result);
            result.borrow_mut()._parent_nodes = vec![a.clone(), b.clone()];
            result.borrow_mut()._back_propagation = Some(Box::new(|child| {
                let parents = &child.borrow()._parent_nodes;
                let scalar_val = parents[1].borrow()._vals[0];
                for (i, &child_grad) in child.borrow()._grad.iter().enumerate() {
                    parents[0].borrow_mut()._grad[i] += child_grad * scalar_val;
                    let value = parents[0].borrow()._vals[i];
                    parents[1].borrow_mut()._grad[0] += child_grad * value;
                }
            }));
            result
        }
        // Matrix multiplication
        else if self.shape().1 == b.shape().0 {
            // (m x n) * (n x p) = (m x p)
            let (m, n) = self.shape();
            let (_, p) = b.shape();

            let lhs = &self.borrow()._vals;
            let rhs = &b.borrow()._vals;

            let result = (0..m * p)
                .map(|i| {
                    let row = i / p;
                    let col = i % p;
                    let lhs_row = lhs.iter().skip(row * n).take(n);
                    let rhs_col = rhs.iter().skip(col).step_by(p);

                    lhs_row.zip(rhs_col).map(|(&r, &c)| r * c).sum()
                })
                .collect();
            let result = CalcNode::new_from_shape((m, p), result);
            result.borrow_mut()._parent_nodes = vec![self.clone(), b.clone()];
            result.borrow_mut()._back_propagation = Some(Box::new(|child| {
                let parents = &child.borrow()._parent_nodes;
                let (_m, n) = parents[0].shape();
                let (_, p) = parents[1].shape();

                for (i, &child_grad) in child.borrow()._grad.iter().enumerate() {
                    let row = i / p;
                    let col = i % p;
                    {
                        // Need to avoid borrowing mutables in the case when lhs = rhs
                        let mut rhs = vec![0.; parents[1].len()];
                        {
                            // RHS derivative
                            let lhs = &parents[0].borrow()._vals;
                            //let rhs = &mut parents[1].borrow_mut()._grad;
                            let lhs_row = lhs.iter().skip(row * n).take(n);
                            let rhs_col = rhs.iter_mut().skip(col).step_by(p);
                            rhs_col
                                .zip(lhs_row)
                                .for_each(|(d, &v)| *d += v * child_grad);
                        }
                        parents[1]
                            .borrow_mut()
                            ._grad
                            .iter_mut()
                            .zip(rhs.into_iter())
                            .for_each(|(target, new_value)| *target += new_value);
                    }

                    {
                        // Need to avoid borrowing mutables in the case when lhs = rhs
                        let mut lhs = vec![0.; parents[0].len()];
                        {
                        // LHS derivative
                        let rhs = &parents[1].borrow()._vals;
                        let lhs_row = lhs.iter_mut().skip(row * n).take(n);
                        let rhs_col = rhs.iter().skip(col).step_by(p);
                        lhs_row
                            .zip(rhs_col)
                            .for_each(|(d, &v)| *d += v * child_grad);
                        }
                        parents[0]
                            .borrow_mut()
                            ._grad
                            .iter_mut()
                            .zip(lhs.into_iter())
                            .for_each(|(target, new_value)| *target += new_value);
                    }
                }
            }));
            result
        } else {
            panic!(
                "Invalid operands for multiplication. {} and {}",
                self, other
            );
        }
    }
}

/// Subtraction is implemented for all combinations of [CalcNode] and &[CalcNode].
impl Sub<&CalcNode> for CalcNode {
    type Output = CalcNode;

    fn sub(self, rhs: &Self) -> Self::Output {
        &self - rhs
    }
}
impl Sub for &CalcNode {
    type Output = CalcNode;

    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}
impl Sub<CalcNode> for &CalcNode {
    type Output = CalcNode;

    fn sub(self, rhs: CalcNode) -> Self::Output {
        self - &rhs
    }
}
impl Sub for CalcNode {
    type Output = CalcNode;

    fn sub(self, rhs: Self) -> Self::Output {
        &self - &rhs
    }
}

/// Division is implemented for all combinations of [CalcNode] and &[CalcNode].
/// 
/// If the RHS is a sclar every LHS value is divided by that scalar. Otherwise, an element-wise division is performed.
impl Div<&CalcNode> for CalcNode {
    type Output = CalcNode;

    fn div(self, rhs: &Self) -> Self::Output {
        &self / rhs
    }
}
impl Div for &CalcNode {
    type Output = CalcNode;

    fn div(self, rhs: Self) -> Self::Output {
        if rhs.node_type() == NodeType::Scalar {
            self * rhs.inv()
        }
        else{
            self.element_wise_div(rhs)
        }
    }
}
impl Div for CalcNode {
    type Output = CalcNode;

    fn div(self, rhs: Self) -> Self::Output {
        &self / &rhs
    }
}
impl Div<CalcNode> for &CalcNode {
    type Output = CalcNode;

    fn div(self, rhs: CalcNode) -> Self::Output {
        self / &rhs
    }
}

impl Neg for CalcNode {
    type Output = CalcNode;

    fn neg(self) -> Self::Output {
        -&self
    }
}
impl Neg for &CalcNode {
    type Output = CalcNode;

    fn neg(self) -> Self::Output {
        self * CalcNode::new_scalar(-1.)
    }
}

impl CalcNode {
    /// Inverts all values.
    pub fn inv(&self) -> CalcNode {
        self.pow(&Self::new_scalar(-1.))
    }
}

impl CalcNode {
    /// Exponentiates all values.    
    pub fn exp(&self) -> CalcNode {
        let result = Self::new_from_shape(
            self.borrow()._shape,
            self.borrow()._vals.iter().map(|v| v.exp()).collect(),
        );
        result.borrow_mut()._parent_nodes = vec![self.clone()];
        result.borrow_mut()._back_propagation = Some(Box::new(|child| {
            child.copy_parents()[0]
                .borrow_mut()
                ._grad
                .iter_mut()
                .zip(child.borrow()._vals.iter().zip(child.borrow()._grad.iter()))
                .for_each(|(pg, (cv, cg))| *pg += cg * cv);
        }));
        result
    }
}

// Log
impl CalcNode {
    /// Applies the natural logarithm to all values.    
    pub fn log(&self) -> CalcNode {
        let result = Self::new_from_shape(
            self.borrow()._shape,
            self.borrow()._vals.iter().map(|v| v.ln()).collect(),
        );
        result.borrow_mut()._parent_nodes = vec![self.clone()];
        result.borrow_mut()._back_propagation = Some(Box::new(|child| {
            let parent = &child.borrow()._parent_nodes[0];
            for i in 0..parent.len() {
                let gradient = child.borrow()._grad[i] / parent.borrow()._vals[i];
                parent.borrow_mut()._grad[i] += gradient;
            }
        }));
        result
    }
}

impl CalcNode {
    /// Applies the power function to all values.    
    pub fn pow(&self, power: &CalcNode) -> CalcNode {
        assert!(power.len() == 1);
        let p = power.value_indexed(0);
        let result = Self::new_from_shape(
            self.borrow()._shape,
            self.borrow()._vals.iter().map(|v| v.powf(p)).collect(),
        );
        result.borrow_mut()._parent_nodes = vec![self.clone(), power.clone()];
        result.borrow_mut()._back_propagation = Some(Box::new(|child| {
            let parents = &child.borrow()._parent_nodes;
            let base = &parents[0];
            let power = &parents[1];
            let power_val = power.value_indexed(0);
            for i in 0..base.len() {
                let base_val = base.value_indexed(i);
                let child_val = child.value_indexed(i);
                let child_grad = child.gradient_indexed(i);
                let gradient = child_grad * power_val * base_val.powf(power_val - 1.);
                base.borrow_mut()._grad[i] += gradient;
                let gradient = child_grad * base_val.ln() * child_val;
                power.borrow_mut()._grad[0] += gradient;
            }
        }));
        result
    }
}

impl CalcNode {
    /// Multiplies the two objects element-wise.
    /// 
    /// Multiplication by scalar and the dot product are provided by [CalcNode::mul].
    pub fn element_wise_mul(&self, other: &Self) -> CalcNode {
        let result = self
            .borrow()
            ._vals
            .iter()
            .zip(other.borrow()._vals.iter())
            .map(|(a, b)| a * b)
            .collect();
        let result = CalcNode::new_from_shape(self.shape(), result);
        result.borrow_mut()._parent_nodes = vec![self.clone(), other.clone()];
        result.borrow_mut()._back_propagation = Some(Box::new(|child| {
            let parents = &child.borrow()._parent_nodes;
            for (i, &child_grad) in child.borrow()._grad.iter().enumerate() {
                let val0 = parents[0].borrow()._vals[i];
                let val1 = parents[1].borrow()._vals[i];
                parents[0].borrow_mut()._grad[i] += child_grad * val1;
                parents[1].borrow_mut()._grad[i] += child_grad * val0;
            }
        }));
        result
    }

    fn element_wise_div(&self, other: &Self) -> CalcNode {
        self.element_wise_mul(&other.inv())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    //use assert_approx_eq::assert_approx_eq;

    #[test]
    fn addition_of_two_scalars() {
        let inp1 = CalcNode::new_scalar(1.);
        let inp2 = CalcNode::new_scalar(2.);
        let mut out = &inp1 + &inp2;
        assert_eq!(out.value_indexed(0), 3.);
        out.back_propagation();
        assert_eq!(out.gradient_indexed(0), 1.);
        assert_eq!(inp1.gradient_indexed(0), 1.);
        assert_eq!(inp2.gradient_indexed(0), 1.);
    }

    #[test]
    fn addition_of_scalar_to_itself() {
        let inp = CalcNode::new_scalar(1.);
        let mut out = &inp + &inp;
        assert_eq!(out.value_indexed(0), 2.);
        out.back_propagation();
        assert_eq!(out.gradient_indexed(0), 1.);
        assert_eq!(inp.gradient_indexed(0), 2.);
    }
    
        #[test]
        fn addition_of_vector_to_itself() {
            let inp = CalcNode::new_col_vector(vec![1.,1.]);
            let mut out = &inp + &inp;
            assert_eq!(out.value_indexed(0), 2.);
            assert_eq!(out.value_indexed(1), 2.);
            out.back_propagation();
            assert_eq!(out.gradient_indexed(0), 1.);
            assert_eq!(out.gradient_indexed(1), 1.);
            assert_eq!(inp.gradient_indexed(0), 2.);
            assert_eq!(inp.gradient_indexed(1), 2.);
        }

    #[test]
    fn addition_of_vector_and_scalar() {
        let inp1 = CalcNode::new_col_vector(vec![1., 2.]);
        let inp2 = CalcNode::new_scalar(3.);
        let expected_value = &[4., 5.];
        let expected_derivative1 = &[1., 1.];
        let expected_derivative2 = &[1.];
        let mut out = &inp1 + &inp2;
        assert_eq!(out.copy_vals(), expected_value);
        out.back_propagation();
        assert_eq!(out.copy_grad(), &[1., 1.]);
        assert_eq!(inp1.copy_grad(), expected_derivative1);
        assert_eq!(inp2.copy_grad(), expected_derivative2);
    }

    #[test]
    fn addition_of_two_vectors() {
        let inp1 = CalcNode::new_col_vector(vec![1., 2.]);
        let inp2 = CalcNode::new_col_vector(vec![3., 4.]);
        let expected_value = &[4., 6.];
        let expected_derivative1 = &[1., 1.];
        let expected_derivative2 = &[1., 1.];
        let mut out = &inp1 + &inp2;
        assert_eq!(out.copy_vals(), expected_value);
        out.back_propagation();
        assert_eq!(out.copy_grad(), &[1., 1.]);
        assert_eq!(inp1.copy_grad(), expected_derivative1);
        assert_eq!(inp2.copy_grad(), expected_derivative2);
    }

    #[test]
    fn multiplication_of_scalar_to_itself() {
        let inp = CalcNode::new_scalar(3.);
        let mut out = &inp * &inp;
        assert_eq!(out.value_indexed(0), 9.);
        out.back_propagation();
        assert_eq!(out.gradient_indexed(0), 1.);
        assert_eq!(inp.gradient_indexed(0), 6.);
    }

    #[test]
    fn elementwise_multiplication_of_vector_to_itself() {
        let inp = CalcNode::new_col_vector(vec![3., 4.]);
        let mut out = inp.element_wise_mul(&inp);
        assert_eq!(out.value_indexed(0), 9.);
        assert_eq!(out.value_indexed(1), 16.);
        out.back_propagation();
        assert_eq!(out.gradient_indexed(0), 1.);
        assert_eq!(out.gradient_indexed(1), 1.);
        assert_eq!(inp.gradient_indexed(0), 6.);
        assert_eq!(inp.gradient_indexed(1), 8.);
    }

    #[test]
    fn multiplication_of_matrix_to_itself() {
        let inp = CalcNode::new_from_shape((2, 2), vec![1., 0., 0., 1.]);
        let mut out = &inp * &inp;
        assert_eq!(out.value_indexed(0), 1.);
        assert_eq!(out.value_indexed(1), 0.);
        assert_eq!(out.value_indexed(2), 0.);
        assert_eq!(out.value_indexed(3), 1.);
        out.back_propagation();
        assert_eq!(out.gradient_indexed(0), 1.);
        assert_eq!(out.gradient_indexed(1), 1.);
        assert_eq!(out.gradient_indexed(2), 1.);
        assert_eq!(out.gradient_indexed(3), 1.);
        assert_eq!(inp.gradient_indexed(0), 2.);
        assert_eq!(inp.gradient_indexed(1), 2.);
        assert_eq!(inp.gradient_indexed(2), 2.);
        assert_eq!(inp.gradient_indexed(3), 2.);
    }

    #[test]
    fn multiplication_of_vector_and_scalar() {
        let inp1 = CalcNode::new_col_vector(vec![1., 2.]);
        let inp2 = CalcNode::new_scalar(3.);
        let expected_value = &[3., 6.];
        let expected_derivative1 = &[3., 3.];
        let expected_derivative2 = &[3.];
        let mut out = &inp1 * &inp2;
        assert_eq!(out.copy_vals(), expected_value);
        out.back_propagation();
        assert_eq!(out.copy_grad(), &[1., 1.]);
        assert_eq!(inp1.copy_grad(), expected_derivative1);
        assert_eq!(inp2.copy_grad(), expected_derivative2);
    }

    #[test]
    fn multiplication_of_two_vectors() {
        let inp1 = CalcNode::new_row_vector(vec![1., 2.]);
        let inp2 = CalcNode::new_col_vector(vec![3., 4.]);
        let expected_value = &[11.];
        let expected_derivative1 = &[3., 4.];
        let expected_derivative2 = &[1., 2.];
        let mut out = &inp1 * &inp2;
        assert_eq!(out.copy_vals(), expected_value);
        out.back_propagation();
        assert_eq!(out.copy_grad(), &[1.]);
        assert_eq!(inp1.copy_grad(), expected_derivative1);
        assert_eq!(inp2.copy_grad(), expected_derivative2);
    }

    #[test]
    fn matrix_multiplication_with_vector() {
        let inp1 = CalcNode::new_from_shape((2, 2), vec![1., 2., 3., 4.]);
        let inp2 = CalcNode::new_col_vector(vec![5., 6.]);
        let expected_value = &[17., 39.];
        let expected_derivative1 = &[5., 6., 5., 6.];
        let expected_derivative2 = &[4., 6.];
        let mut out = &inp1 * &inp2;
        assert_eq!(out.copy_vals(), expected_value);
        out.back_propagation();
        assert_eq!(out.copy_grad(), &[1., 1.]);
        assert_eq!(inp1.copy_grad(), expected_derivative1);
        assert_eq!(inp2.copy_grad(), expected_derivative2);
    }

    #[test]
    fn matrix_multiplication_square_matrices() {
        let inp1 = CalcNode::new_from_shape((2, 2), vec![1., 2., 3., 4.]);
        let inp2 = CalcNode::new_from_shape((2, 2), vec![5., 6., 7., 8.]);
        let expected_value = &[19., 22., 43., 50.];
        let expected_derivative1 = &[11., 15., 11., 15.];
        let expected_derivative2 = &[4., 4., 6., 6.];
        let mut out = &inp1 * &inp2;
        assert_eq!(out.copy_vals(), expected_value);
        out.back_propagation();
        assert_eq!(out.copy_grad(), &[1., 1., 1., 1.]);
        assert_eq!(inp1.copy_grad(), expected_derivative1);
        assert_eq!(inp2.copy_grad(), expected_derivative2);
    }

    #[test]
    fn matrix_multiplication_non_square_matrices() {
        let inp1 = CalcNode::new_from_shape((2, 3), vec![1., 2., 3., 4., 5., 6.]);
        let inp2 = CalcNode::new_from_shape((3, 2), vec![7., 8., 9., 10., 11., 12.]);
        let expected_value = &[58., 64., 139., 154.];
        let expected_derivative1 = &[15., 19., 23., 15., 19., 23.];
        let expected_derivative2 = &[5., 5., 7., 7., 9., 9.];
        let mut out = &inp1 * &inp2;
        assert_eq!(out.copy_vals(), expected_value);
        out.back_propagation();
        assert_eq!(out.copy_grad(), &[1., 1., 1., 1.]);
        assert_eq!(inp1.copy_grad(), expected_derivative1);
        assert_eq!(inp2.copy_grad(), expected_derivative2);
    }
}
