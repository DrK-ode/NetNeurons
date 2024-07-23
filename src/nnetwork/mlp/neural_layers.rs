use std::{fmt::Display, iter};

use crate::nnetwork::calculation_nodes::{TensorShape, TensorShared};

use super::neural_traits::{Forward, Layer, Parameters};

pub struct LinearLayer {
    _w: TensorShared,
    _b: Option<TensorShared>,
}

impl LinearLayer {
    pub fn from_rand(n_in: usize, n_out: usize, biased: bool) -> LinearLayer {
        LinearLayer {
            _w: TensorShared::from_random((n_out, n_in, 1)),
            _b: if biased {
                Some(TensorShared::from_random((n_out, 1, 1)))
            } else {
                None
            },
        }
    }
    pub fn from_tensors(w: TensorShared, b: Option<TensorShared>) -> LinearLayer {
        assert!(
            w.len() > 0 && (b.is_none() || b.as_ref().unwrap().len() > 0),
            "Cannot create layer from empty tensor."
        );
        if let Some(b) = &b {
            assert_eq!(
                w.shape().0,
                b.shape().0,
                "Bias tensor must have equal number of rows as weight tensor."
            );
        }
        LinearLayer { _w: w, _b: b }
    }
}

impl Display for LinearLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LinearLayer: [weights: {}", self._w)?;
        if self._b.is_some() {
            write!(f, ", biases: {}", self._w)?;
        }
        writeln!(f, "]")
    }
}

impl Forward for LinearLayer {
    fn forward(&self, prev: &TensorShared) -> TensorShared {
        if self._b.is_some() {
            &self._w * prev + self._b.as_ref().unwrap()
        } else {
            &self._w * prev
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
}

pub struct FunctionLayer {
    _func: &'static dyn Fn(&TensorShared) -> TensorShared,
    _label: String,
}

impl FunctionLayer {
    pub fn new(
        f: &'static dyn Fn(&TensorShared) -> TensorShared,
        label: &str,
    ) -> FunctionLayer {
        FunctionLayer {
            _func: f,
            _label: label.into(),
        }
    }
}

impl Display for FunctionLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "FunctionLayer: [{}]", self._label)
    }
}

impl Forward for FunctionLayer {
    fn forward(&self, inp: &TensorShared) -> TensorShared {
        (self._func)(inp)
    }
}
impl Parameters for FunctionLayer {}
impl Layer for FunctionLayer {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nnetwork::calculation_nodes::NetworkCalculation;

    #[test]
    fn unbiased_layer_forward() {
        let layer = LinearLayer::from_tensors(
            TensorShared::from_vector(vec![1., 2., 3., 4.], (2, 2, 1)),
            None,
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
