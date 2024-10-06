use std::{fmt::Display, iter};

use crate::nnetwork::{CalcNodeShared, NodeShape};

use super::parameters::Parameters;

pub trait Layer: Parameters + Display {
    fn shape(&self) -> Option<NodeShape> {
        None
    }

    fn forward(&self, inp: &CalcNodeShared) -> CalcNodeShared;

    fn layer_name(&self) -> &str;
}

pub struct LinearLayer {
    _w: CalcNodeShared,
    _b: Option<CalcNodeShared>,
    _label: String,
}

impl LinearLayer {
    pub fn from_rand(n_rows: usize, n_cols: usize, biased: bool, label: &str) -> LinearLayer {
        LinearLayer {
            _w: CalcNodeShared::rand_from_shape((n_rows, n_cols)),
            _b: if biased {
                Some(CalcNodeShared::rand_from_shape((n_rows, 1)))
            } else {
                None
            },
            _label: label.to_string(),
        }
    }
    pub fn from_nodes(w: CalcNodeShared, b: Option<CalcNodeShared>, label: &str) -> LinearLayer {
        assert!(
            !w.is_empty() && (b.is_none() || !b.as_ref().unwrap().is_empty()),
            "Cannot create layer from empty tensor."
        );
        if let Some(b) = &b {
            assert_eq!(
                w.shape().0,
                b.shape().0,
                "Bias vector must have equal number of rows as weight matrix."
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

impl Parameters for LinearLayer {
    fn param_iter(&self) -> Box<dyn Iterator<Item = &CalcNodeShared> + '_> {
        let parameters = iter::once(&self._w);
        if self._b.is_some() {
            Box::new(parameters.chain(iter::once(self._b.as_ref().unwrap())))
        } else {
            Box::new(parameters)
        }
    }
}

impl Layer for LinearLayer {
    fn forward(&self, prev: &CalcNodeShared) -> CalcNodeShared {
        if self._b.is_some() {
            &self._w * prev + self._b.as_ref().unwrap()
        } else {
            &self._w * prev
        }
    }

    fn layer_name(&self) -> &str {
        &self._label
    }
}

pub struct ReshapeLayer {
    _shape: NodeShape,
    _label: String,
}

impl ReshapeLayer {
    pub fn new(shape: NodeShape, label: &str) -> Self {
        ReshapeLayer {
            _shape: shape,
            _label: label.to_string(),
        }
    }
}

impl Display for ReshapeLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "ReshapeLayer ({}): [{:?}]", self._label, self._shape)
    }
}

impl Parameters for ReshapeLayer {}

impl Layer for ReshapeLayer {
    fn forward(&self, inp: &CalcNodeShared) -> CalcNodeShared {
        let mut out = inp.clone();
        out.reshape(self._shape);
        out
    }

    fn layer_name(&self) -> &str {
        &self._label
    }
}

#[derive(Clone)]
pub struct FunctionLayer {
    _func: &'static dyn Fn(&CalcNodeShared) -> CalcNodeShared,
    _formula: String,
    _label: String,
}

impl FunctionLayer {
    pub fn new(
        f: &'static dyn Fn(&CalcNodeShared) -> CalcNodeShared,
        formula: &str,
        label: &str,
    ) -> FunctionLayer {
        FunctionLayer {
            _func: f,
            _formula: formula.into(),
            _label: label.into(),
        }
    }

    pub fn sigmoid(inp: &CalcNodeShared) -> CalcNodeShared {
        (CalcNodeShared::filled_from_shape(inp.shape(), vec![1.; inp.len()]) + (-inp).exp())
            .pow(&CalcNodeShared::new_scalar(-1.))
    }

    pub fn tanh(inp: &CalcNodeShared) -> CalcNodeShared {
        let one = CalcNodeShared::new_scalar(1.);
        let a = -inp;
        let b = a * CalcNodeShared::new_scalar(2.);
        let exp2 = b.exp();
        //let exp2 = (-inp * CalcNodeShared::new_scalar(2.)).exp();
        (&one - &exp2).element_wise_div(&(&one + &exp2))
    }

    pub fn softmax(inp: &CalcNodeShared) -> CalcNodeShared {
        inp.exp().normalized()
    }
}

impl Display for FunctionLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "FunctionLayer ({}): [{}]", self._formula, self._label)
    }
}

impl Parameters for FunctionLayer {}

impl Layer for FunctionLayer {
    fn forward(&self, inp: &CalcNodeShared) -> CalcNodeShared {
        (self._func)(inp)
    }
    fn layer_name(&self) -> &str {
        &self._label
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn unbiased_layer_forward() {
        let layer = LinearLayer::from_nodes(
            CalcNodeShared::filled_from_shape((2, 2), vec![1., 2., 3., 4.]),
            None,
            "TestLayer",
        );
        let inp = CalcNodeShared::new_col_vector(vec![5., 6.]);
        let expected_value = &[17., 39.];
        let expected_derivative1 = &[5., 6., 5., 6.];
        let expected_derivative2 = &[4., 6.];
        let out = layer.forward(&inp);
        assert_eq!(out.copy_vals(), expected_value);
        out.back_propagation();
        assert_eq!(out.copy_grad(), &[1., 1.]);
        assert_eq!(
            layer
                .param_iter()
                .flat_map(|p| p.copy_grad())
                .collect::<Vec<_>>(),
            expected_derivative1
        );
        assert_eq!(inp.copy_grad(), expected_derivative2);
    }

    #[test]
    fn biased_layer_forward() {
        let layer = LinearLayer::from_nodes(
            CalcNodeShared::filled_from_shape((2, 2), vec![1., 2., 3., 4.]),
            Some(CalcNodeShared::new_col_vector(vec![7., 8.])),
            "TestLayer",
        );
        let inp = CalcNodeShared::new_col_vector(vec![5., 6.]);
        let expected_value = &[17. + 7., 39. + 8.];
        let expected_derivative1 = &[vec![5., 6., 5., 6.], vec![1., 1.]];
        let expected_derivative2 = &[4., 6.];
        let out = layer.forward(&inp);
        assert_eq!(out.copy_vals(), expected_value);
        out.back_propagation();
        assert_eq!(out.copy_grad(), &[1., 1.]);
        assert_eq!(
            layer
                .param_iter()
                .map(|p| p.copy_grad())
                .collect::<Vec<_>>(),
            expected_derivative1
        );
        assert_eq!(inp.copy_grad(), expected_derivative2);
    }

    #[test]
    fn tanh_forward() {
        let layer = FunctionLayer::new(&FunctionLayer::tanh, "tanh", "TestLayer");
        let inp = CalcNodeShared::new_col_vector(vec![-10., -2., -1., 0., 1., 2., 10.]);
        let expected_value = &[-1., -0.9640276, -0.7615942, 0., 0.7615942, 0.9640276, 1.];
        let out = layer.forward(&inp);
        for (value, expected_value) in out.copy_vals().iter().zip(expected_value) {
            assert_approx_eq!(value, expected_value);
        }
    }
}
