use std::{fmt::Display, iter};

use crate::nnetwork::{calculation_nodes::TensorShared, TensorShape};

use super::layer_traits::{Forward, Layer, Parameters};

pub struct LinearLayer {
    _w: TensorShared,
    _b: Option<TensorShared>,
    _label: String,
}

impl LinearLayer {
    pub fn from_rand(n_rows: usize, n_cols: usize, biased: bool, label: &str) -> LinearLayer {
        LinearLayer {
            _w: TensorShared::from_random((n_rows, n_cols, 1)),
            _b: if biased {
                Some(TensorShared::from_random((n_rows, 1, 1)))
            } else {
                None
            },
            _label: label.to_string(),
        }
    }
    pub fn from_tensors(w: TensorShared, b: Option<TensorShared>, label: &str) -> LinearLayer {
        assert!(
            !w.is_empty() && (b.is_none() || !b.as_ref().unwrap().is_empty()),
            "Cannot create layer from empty tensor."
        );
        if let Some(b) = &b {
            assert_eq!(
                w.shape().0,
                b.shape().0,
                "Bias tensor must have equal number of rows as weight tensor."
            );
        }
        LinearLayer {
            _w: w,
            _b: b,
            _label: label.to_string(),
        }
    }
}

impl Display for LinearLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LinearLayer ({}): [weights: {}", self._label, self._w)?;
        if self._b.is_some() {
            write!(f, ", biases: {}", self._w)?;
        }
        writeln!(f, "]")
    }
}

impl Forward for LinearLayer {
    fn forward(&self, prev: &TensorShared) -> TensorShared {
        if self._b.is_some() {
            self._w.dot(prev) + self._b.as_ref().unwrap()
        } else {
            self._w.dot(prev)
        }
    }
}

impl Parameters for LinearLayer {
    fn parameters(&self) -> Box<dyn Iterator<Item = &TensorShared> + '_> {
        let parameters = iter::once(&self._w);
        if self._b.is_some() {
            Box::new(parameters.chain(iter::once(self._b.as_ref().unwrap())))
        } else {
            Box::new(parameters)
        }
    }
}

impl Layer for LinearLayer {
    fn shape(&self) -> Option<TensorShape> {
        Some(self._w.shape())
    }

    fn layer_name(&self) -> &str {
        &self._label
    }
}

pub struct ReshapeLayer {
    _shape: TensorShape,
    _label: String,
}
impl ReshapeLayer {
    pub fn new(shape: TensorShape, label: &str) -> Self {
        ReshapeLayer {
            _shape: shape,
            _label: label.to_string(),
        }
    }
}
impl Forward for ReshapeLayer {
    fn forward(&self, inp: &TensorShared) -> TensorShared {
        let mut out = inp.clone();
        out.reshape(self._shape);
        out
    }
}
impl Display for ReshapeLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "ReshapeLayer ({}): [{:?}]", self._label, self._shape)
    }
}
impl Parameters for ReshapeLayer {}
impl Layer for ReshapeLayer {
    fn layer_name(&self) -> &str {
        &self._label
    }
}

#[derive(Clone)]
pub struct FunctionLayer {
    _func: &'static dyn Fn(&TensorShared) -> TensorShared,
    _formula: String,
    _label: String,
}

impl FunctionLayer {
    pub fn new(f: &'static dyn Fn(&TensorShared) -> TensorShared, formula: &str, label: &str) -> FunctionLayer {
        FunctionLayer {
            _func: f,
            _formula: formula.into(),
            _label: label.into(),
        }
    }

    pub fn sigmoid(inp: &TensorShared) -> TensorShared {
        (TensorShared::from_vector(vec![1.; inp.len()], inp.shape()) + (-inp).exp())
            .pow(&TensorShared::from_scalar(-1.))
    }

    pub fn tanh(inp: &TensorShared) -> TensorShared {
        let one = TensorShared::from_vector(vec![1.; inp.len()], inp.shape());
        let exp2 = (inp * TensorShared::from_scalar(2.)).exp();
        (&one - &exp2) / (&one + &exp2)
    }

    pub fn softmax(inp: &TensorShared) -> TensorShared {
        inp.exp().normalized()
    }
}

impl Display for FunctionLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "FunctionLayer ({}): [{}]", self._formula, self._label)
    }
}

impl Forward for FunctionLayer {
    fn forward(&self, inp: &TensorShared) -> TensorShared {
        (self._func)(inp)
    }
}
impl Parameters for FunctionLayer {}
impl Layer for FunctionLayer {
    fn layer_name(&self) -> &str {
        &self._label
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nnetwork::calculation_nodes::NetworkCalculation;

    #[test]
    fn unbiased_layer_forward() {
        let layer = LinearLayer::from_tensors(
            TensorShared::from_vector(vec![1., 2., 3., 4.], (2, 2, 1)),
            None,
            "TestLayer",
        );
        let inp = TensorShared::from_vector(vec![5., 6.], (2, 1, 1));
        let expected_value = &[17., 39.];
        let expected_derivative1 = &[&[5., 6., 5., 6.]];
        let expected_derivative2 = &[4., 6.];
        let out = layer.forward(&inp);
        let calc = NetworkCalculation::new(&out);
        calc.evaluate();
        assert_eq!(out.value_as_col_vector().unwrap(), expected_value);
        calc.back_propagation();
        assert_eq!(out.derivative_as_col_vector().unwrap(), &[1., 1.]);
        assert_eq!(
            layer
                .parameters()
                .map(|p| p.derivative())
                .collect::<Vec<_>>(),
            expected_derivative1
        );
        assert_eq!(inp.derivative(), expected_derivative2);
    }

    #[test]
    fn biased_layer_forward() {
        let layer = LinearLayer::from_tensors(
            TensorShared::from_vector(vec![1., 2., 3., 4.], (2, 2, 1)),
            Some(TensorShared::from_vector(vec![7., 8.], (2, 1, 1))),
            "TestLayer",
        );
        let inp = TensorShared::from_vector(vec![5., 6.], (2, 1, 1));
        let expected_value = &[17. + 7., 39. + 8.];
        let expected_derivative1 = &[vec![5., 6., 5., 6.], vec![1., 1.]];
        let expected_derivative2 = &[4., 6.];
        let out = layer.forward(&inp);
        let calc = NetworkCalculation::new(&out);
        calc.evaluate();
        assert_eq!(out.value_as_col_vector().unwrap(), expected_value);
        calc.back_propagation();
        assert_eq!(out.derivative_as_col_vector().unwrap(), &[1., 1.]);
        assert_eq!(
            layer
                .parameters()
                .map(|p| p.derivative())
                .collect::<Vec<_>>(),
            expected_derivative1
        );
        assert_eq!(inp.derivative(), expected_derivative2);
    }
}
