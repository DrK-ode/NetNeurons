use std::{fmt::Display, ops::Deref, time::Instant};

use rand::Rng;

use crate::nnetwork::{
    calculation_nodes::{FloatType, NetworkCalculation, TensorShared},
    TensorShape,
};

use super::neural_traits::{Layer, Parameters};

pub struct MultiLayer {
    _layers: Vec<Box<dyn Layer>>,
    _forward_calc: Option<(NetworkCalculation, TensorShared)>,
    _train_calc: Option<(NetworkCalculation, Vec<(TensorShared, TensorShared)>)>,
}

impl MultiLayer {
    pub fn from_empty() -> MultiLayer {
        MultiLayer {
            _layers: Vec::new(),
            _forward_calc: None,
            _train_calc: None,
        }
    }

    pub fn add_layer(&mut self, layer: Box<dyn Layer>) {
        self._layers.push(layer);
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

    pub fn least_squares(inp: &TensorShared, truth: &TensorShared) -> TensorShared {
        (inp - truth).powf(2.).sum()
    }

    // Assumes the input can be treated as a probability distribution and that the truth is a one-hot vector
    pub fn neg_log_likelihood(inp: &TensorShared, truth: &TensorShared) -> TensorShared {
        -(inp * truth).sum().log()
    }

    pub fn define_forward(&mut self, inp_shape: TensorShape) {
        let inp = TensorShared::from_shape(inp_shape);
        let out = self.forward_internal(&inp);
        self._forward_calc = Some((NetworkCalculation::new(&out), inp));
    }

    pub fn forward(&self, inp: &TensorShared) -> TensorShared {
        assert!(
            self._forward_calc.is_some(),
            "Define forward calcuation first"
        );
        let (calc, input) = self._forward_calc.as_ref().unwrap();
        input
            .deref()
            .borrow_mut()
            .set_value(inp.borrow().value().to_vec());
        let out = calc.evaluate();
        TensorShared::from_vector(out.value(), out.shape())
    }

    fn forward_internal(&self, inp: &TensorShared) -> TensorShared {
        let mut out = inp.clone();
        for l in &self._layers {
            out = l.forward(&out);
        }
        out
    }

    pub fn define_training(
        &mut self,
        n_correlations: usize,
        inp_shape: TensorShape,
        out_shape: TensorShape,
        regularization: Option<FloatType>,
        loss_func: &'static dyn Fn(&TensorShared, &TensorShared) -> TensorShared,
    ) {
        let timer = Instant::now();
        if let Some(regularization) = regularization {
            if regularization <= 0. {
                panic!("Regularization coefficient must be positive.");
            }
        }
        let inputs = (0..n_correlations)
            .map(|_| {
                (
                    TensorShared::from_shape(inp_shape),
                    TensorShared::from_shape(out_shape),
                )
            })
            .collect::<Vec<_>>();

        let mut loss = inputs
            .iter()
            .map(|(inp, truth)| (loss_func)(&self.forward_internal(inp), truth))
            .sum::<TensorShared>()
            / TensorShared::from_scalar(n_correlations as FloatType);

        if regularization.is_some() {
            let regularization = TensorShared::from_scalar(regularization.unwrap());
            let n_param = TensorShared::from_scalar(self.parameters().count() as FloatType);
            // Mean of the sum of the squares of all parameters
            let reg_loss = self.parameters().map(|p| p.powf(2.)).sum::<TensorShared>()
                * regularization
                / n_param;
            loss = loss + reg_loss;
        };

        println!(
            "Defining calcualtion took {} µs",
            timer.elapsed().as_micros()
        );

        let timer = Instant::now();
        self._train_calc = Some((NetworkCalculation::new(&loss), inputs));
        println!(
            "Topological sorting took {} µs",
            timer.elapsed().as_micros()
        );
    }

    fn load_correlations(&mut self, inp: &[(TensorShared, TensorShared)]) {
        assert!(
            self._train_calc.is_some(),
            "Calculation must be defined before loading input values."
        );
        // Copy values over to the input tensors
        assert_eq!(
            inp.len(),
            self._train_calc.as_ref().unwrap().1.len(),
            "Calculation must be redefined before changing the number of inputs."
        );
        if let Some((_, ref mut inputs)) = self._train_calc {
            inputs.iter_mut().zip(inp.iter()).for_each(|(a, b)| {
                a.0.borrow_mut().set_value(b.0.borrow().value().to_vec());
                a.1.borrow_mut().set_value(b.1.borrow().value().to_vec());
            })
        }
    }

    pub fn train(
        &mut self,
        inp: &[(TensorShared, TensorShared)],
        learning_rate: FloatType,
        verbose: bool,
    ) -> TensorShared {
        self.load_correlations(inp);
        let calc = &self._train_calc.as_ref().unwrap().0;

        let timer = Instant::now();
        let loss = calc.evaluate();
        if verbose {
            println!(
                "Performing calculation took {} µs",
                timer.elapsed().as_micros()
            )
        }

        let timer = Instant::now();
        calc.back_propagation();
        if verbose {
            println!("Back propagation took {} µs", timer.elapsed().as_micros())
        }

        self.decend_grad(learning_rate);

        loss
    }

    fn decend_grad(&self, learning_rate: FloatType) {
        self.parameters().for_each(|p| p.decend_grad(learning_rate));
    }
}

impl Display for MultiLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "MLP: [")?;
        for layer in &self._layers {
            layer.fmt(f)?;
        }
        writeln!(f, "]")
    }
}

impl Parameters for MultiLayer {
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
        let mut mlp = MultiLayer::from_empty();
        mlp.add_layer(Box::new(LinearLayer::from_tensors(
            TensorShared::from_vector(vec![1., 1., 1., 1., 1., 1.], (3, 2, 1)),
            None,
        )));
        mlp.add_layer(Box::new(LinearLayer::from_tensors(
            TensorShared::from_vector(vec![1., 1., 1., 1., 1., 1., 1., 1., 1.], (3, 3, 1)),
            None,
        )));
        mlp.add_layer(Box::new(LinearLayer::from_tensors(
            TensorShared::from_vector(vec![1., 1., 1., 1., 1., 1.], (2, 3, 1)),
            None,
        )));
        let inp = TensorShared::from_vector(vec![1., 2.], (2, 1, 1));
        mlp.define_forward(inp.shape());
        let output = mlp.forward(&inp);
        assert_eq!(output.value_as_col_vector().unwrap(), vec![27., 27.]);
    }

    #[test]
    fn mlp_function_layer() {
        fn hej(t: &TensorShared) -> TensorShared {
            t.powf(1.)
        }
        let mut mlp = MultiLayer::from_empty();
        mlp.add_layer(Box::new(FunctionLayer::new(&hej, "hej")));
        let inp = TensorShared::from_vector(vec![1., 2.], (2, 1, 1));
        mlp.define_forward(inp.shape());
        let output = mlp.forward(&inp);
        assert_eq!(output.value_as_col_vector().unwrap(), vec![1., 2.]);
    }

    #[test]
    fn mlp_sigmoid_layer() {
        let mut mlp = MultiLayer::from_empty();
        mlp.add_layer(Box::new(FunctionLayer::new(
            &FunctionLayer::sigmoid,
            "sigmoid",
        )));
        let inp = TensorShared::from_vector(vec![1., 2.], (2, 1, 1));
        mlp.define_forward(inp.shape());
        let output = mlp.forward(&inp);
        assert_eq!(
            output.value_as_col_vector().unwrap(),
            vec![1. / (1. + (-1f64).exp()), 1. / (1. + (-2f64).exp())]
        );
    }

    #[test]
    fn mlp_softmax_layer() {
        let mut mlp = MultiLayer::from_empty();
        mlp.add_layer(Box::new(FunctionLayer::new(
            &FunctionLayer::softmax,
            "softmax",
        )));
        let inp = TensorShared::from_vector(vec![1., 2.], (2, 1, 1));
        mlp.define_forward(inp.shape());
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
