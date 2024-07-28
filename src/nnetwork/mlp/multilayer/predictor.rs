use std::fmt::Display;

use rand::Rng;

use crate::nnetwork::{
    calculation_nodes::{NetworkCalculation, TensorShared},
    Forward, TensorShape,
};

use crate::nnetwork::mlp::layer_traits::{Layer, Parameters};

pub struct Predictor {
    _layers: Vec<Box<dyn Layer>>,
    _fw_inp: TensorShared,
    _calc: NetworkCalculation,
}

impl Predictor {
    pub fn new(inp_shape: TensorShape, layers: Vec<Box<dyn Layer>>) -> Self {
        let inp = TensorShared::from_shape(inp_shape);
        let calc = Self::define_forward_calc(&inp, &layers);
        Predictor {
            _layers: layers,
            _fw_inp: inp,
            _calc: calc,
        }
    }

    fn define_forward_calc(inp: &TensorShared, layers: &[Box<dyn Layer>]) -> NetworkCalculation {
        let out = layers.iter().fold(inp.clone(), |out, l| l.forward(&out));
        NetworkCalculation::new(&out)
    }

    pub fn collapse(inp: &TensorShared) -> TensorShared {
        let mut vec = vec![0.; inp.len()];
        let mut rnd = rand::thread_rng().gen_range(0. ..inp.borrow().value().iter().sum());
        for (i, &v) in inp.borrow().value().iter().enumerate() {
            rnd -= v;
            if rnd <= 0. || i + 1 == inp.len() {
                // Safe-guard against float precision errors
                vec[i] = 1.;
                break;
            }
        }
        TensorShared::from_vector(vec, inp.shape())
    }
}

impl Forward for Predictor {
    fn forward(&self, inp: &TensorShared) -> TensorShared {
        self._fw_inp
            .borrow_mut()
            .set_value(inp.borrow().value().to_vec());
        let out = self._calc.evaluate();
        TensorShared::from_vector(out.value(), out.shape())
    }
}

impl Display for Predictor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Predictor: [")?;
        for layer in &self._layers {
            layer.fmt(f)?;
        }
        writeln!(f, "]")
    }
}

impl Parameters for Predictor {
    fn parameters(&self) -> Box<dyn Iterator<Item = &TensorShared> + '_> {
        Box::new(self._layers.iter().flat_map(|l| l.parameters()))
    }
}

#[cfg(test)]
mod tests {
    use crate::nnetwork::{FunctionLayer, LinearLayer};

    use super::*;

    #[test]
    fn mlp_forward() {
        let layers: Vec<Box<dyn Layer>> = vec![
            Box::new(LinearLayer::from_tensors(
                TensorShared::from_vector(vec![1., 1., 1., 1., 1., 1.], (3, 2, 1)),
                None,
                "TestLayer",
            )),
            Box::new(LinearLayer::from_tensors(
                TensorShared::from_vector(vec![1., 1., 1., 1., 1., 1., 1., 1., 1.], (3, 3, 1)),
                None,
                "TestLayer",
            )),
            Box::new(LinearLayer::from_tensors(
                TensorShared::from_vector(vec![1., 1., 1., 1., 1., 1.], (2, 3, 1)),
                None,
                "TestLayer",
            )),
        ];
        let inp = TensorShared::from_vector(vec![1., 2.], (2, 1, 1));
        let mlp = Predictor::new(inp.shape(), layers);
        let output = mlp.forward(&inp);
        assert_eq!(output.value_as_col_vector().unwrap(), vec![27., 27.]);
    }

    #[test]
    fn mlp_function_layer() {
        fn hej(t: &TensorShared) -> TensorShared {
            t.powf(1.)
        }
        let layers: Vec<Box<dyn Layer>> =
            vec![Box::new(FunctionLayer::new(&hej, "hej", "hej layer"))];
        let inp = TensorShared::from_vector(vec![1., 2.], (2, 1, 1));
        let mlp = Predictor::new(inp.shape(), layers);
        let output = mlp.forward(&inp);
        assert_eq!(output.value_as_col_vector().unwrap(), vec![1., 2.]);
    }

    #[test]
    fn mlp_sigmoid_layer() {
        let layers: Vec<Box<dyn Layer>> = vec![Box::new(FunctionLayer::new(
            &FunctionLayer::sigmoid,
            "sigmoid layer",
            "sigmoid",
        ))];
        let inp = TensorShared::from_vector(vec![1., 2.], (2, 1, 1));
        let mlp = Predictor::new(inp.shape(), layers);
        let output = mlp.forward(&inp);
        assert_eq!(
            output.value_as_col_vector().unwrap(),
            vec![1. / (1. + (-1f64).exp()), 1. / (1. + (-2f64).exp())]
        );
    }

    #[test]
    fn mlp_softmax_layer() {
        let layers: Vec<Box<dyn Layer>> = vec![Box::new(FunctionLayer::new(
            &FunctionLayer::softmax,
            "softmax",
            "TestLayer",
        ))];
        let inp = TensorShared::from_vector(vec![1., 2.], (2, 1, 1));
        let mlp = Predictor::new(inp.shape(), layers);
        let output = mlp.forward(&inp);
        assert_eq!(
            output.value_as_col_vector().unwrap(),
            vec![
                1f64.exp() / (1f64.exp() + 2f64.exp()),
                2f64.exp() / (1f64.exp() + 2f64.exp())
            ]
        );
    }
}
