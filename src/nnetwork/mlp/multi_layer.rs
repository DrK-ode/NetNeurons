use std::{fmt::Display, ops::Deref, time::Instant};

use rand::Rng;

use crate::nnetwork::{
    calculation_nodes::{FloatType, NetworkCalculation, TensorShared},
    TensorShape,
};

use super::{
    layer_traits::{Layer, Parameters},
    Forward, FunctionLayer, LinearLayer,
};

pub struct MultiLayer {
    _embed: Option<LinearLayer>,
    _layers: Vec<Box<dyn Layer>>,
    _forward: Option<(TensorShared, NetworkCalculation)>,
    _train: Option<(Vec<(TensorShared, TensorShared)>, NetworkCalculation)>,
}

impl MultiLayer {
    fn new_blank(
        inp_shape: TensorShape,
        embed_dim: Option<usize>,
        out_shape: TensorShape,
        layers: Vec<Box<dyn Layer>>,
    ) -> Self {
        let (input_rows, block_size, _) = inp_shape;
        let mut ml_layers: Vec<Box<dyn Layer>> = Vec::new();

        // Embedding layer
        let embed = if let Some(embed_dim) = embed_dim {
            let embed = LinearLayer::from_rand(block_size, input_rows, false);

            let mut first_layer_inp_size = out_shape.0;
            for l in &layers {
                if let Some(shape) = l.shape() {
                    first_layer_inp_size = shape.0;
                }
            }
            ml_layers.push(Box::new(LinearLayer::from_rand(
                block_size * embed_dim,
                first_layer_inp_size,
                true,
            )));
            ml_layers.push(Box::new(FunctionLayer::new(&FunctionLayer::tanh, "Tanh")));
            Some(embed)
        } else {
            None
        };
        ml_layers.extend(layers);

        MultiLayer {
            _embed: embed,
            _layers: ml_layers,
            _forward: None,
            _train: None,
        }
    }
    pub fn new_predictor(
        inp_shape: TensorShape,
        embed_dim: Option<usize>,
        out_shape: TensorShape,
        layers: Vec<Box<dyn Layer>>,
    ) -> Self {
        let mut ml = Self::new_blank(inp_shape, embed_dim, out_shape, layers);
        let inp = TensorShared::from_shape(inp_shape);
        let calc = Self::define_forward_calc(&inp, &ml._embed, &ml._layers);
        ml._forward = Some((inp, calc));
        ml
    }

    pub fn new_trainable(
        inp_shape: TensorShape,
        embed_dim: Option<usize>,
        out_shape: TensorShape,
        batch_size: usize,
        layers: Vec<Box<dyn Layer>>,
        regularization: Option<FloatType>,
        loss_func: &'static dyn Fn(&TensorShared, &TensorShared) -> TensorShared,
    ) -> Self {
        let mut ml = MultiLayer::new_blank(inp_shape, embed_dim, out_shape, layers);
        let train_inp = (0..batch_size)
            .map(|_| {
                (
                    TensorShared::from_shape((inp_shape.0, inp_shape.1, 1)),
                    TensorShared::from_shape((out_shape.0, out_shape.1, 1)),
                )
            })
            .collect::<Vec<_>>();
        let train_calc = Self::define_train_calc(&mut ml, &train_inp, regularization, loss_func);
        ml._train = Some((train_inp, train_calc));

        ml
    }

    fn define_forward_calc(
        inp: &TensorShared,
        embed: &Option<LinearLayer>,
        layers: &[Box<dyn Layer>],
    ) -> NetworkCalculation {
        let emb = if let Some(embed) = embed {
            &Self::define_embedding(embed, inp)
        } else {
            inp
        };
        let out = Self::define_layer_forward(emb, layers);
        NetworkCalculation::new(&out)
    }

    fn define_train_calc(
        ml: &mut MultiLayer,
        inp: &[(TensorShared, TensorShared)],
        regularization: Option<FloatType>,
        loss_func: &'static dyn Fn(&TensorShared, &TensorShared) -> TensorShared,
    ) -> NetworkCalculation {
        let out = ml.define_loss(
            &inp.iter()
                .map(|(inp, truth)| {
                    let emb = if let Some(embed) = &ml._embed {
                        &Self::define_embedding(embed, inp) } else { inp };
                    let out = Self::define_layer_forward(emb, &ml._layers);
                    (out, truth.clone())
                })
                .collect::<Vec<_>>(),
            regularization,
            loss_func,
        );

        let timer = Instant::now();
        let calc = NetworkCalculation::new(&out);
        println!(
            "Topological sorting took {} µs",
            timer.elapsed().as_micros()
        );
        calc
    }

    fn define_embedding(embed: &LinearLayer, inp: &TensorShared) -> TensorShared {
        let mut out = embed.forward(inp);
        out.reshape((inp.len(), 1, 1));
        out
    }

    fn define_layer_forward(inp: &TensorShared, layers: &[Box<dyn Layer>]) -> TensorShared {
        let mut out = inp.clone();
        for l in layers {
            out = l.forward(&out);
        }
        out
    }

    fn define_loss(
        &self,
        inp: &[(TensorShared, TensorShared)],
        regularization: Option<FloatType>,
        loss_func: &'static dyn Fn(&TensorShared, &TensorShared) -> TensorShared,
    ) -> TensorShared {
        let timer = Instant::now();
        if let Some(regularization) = regularization {
            if regularization <= 0. {
                panic!("Regularization coefficient must be positive.");
            }
        }

        let mut loss = inp
            .iter()
            .map(|(out, truth)| (loss_func)(out, truth))
            .sum::<TensorShared>()
            * TensorShared::from_scalar(1. / inp.len() as FloatType);

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
            "Defining calculation took {} µs",
            timer.elapsed().as_micros()
        );

        loss
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

    pub fn forward(&self, inp: &TensorShared) -> TensorShared {
        if let Some((fw_inp, calc)) = &self._forward {
            fw_inp
                .deref()
                .borrow_mut()
                .set_value(inp.borrow().value().to_vec());
            let out = calc.evaluate();
            TensorShared::from_vector(out.value(), out.shape())
        } else {
            panic!("Forward calculation not prepared.");
        }
    }

    // Any input tensor will be flattened to a column vector
    fn load_correlations(&mut self, inp: &[(TensorShared, TensorShared)]) {
        if let Some((ref mut train_inp, _)) = &mut self._train {
            // Copy values over to the input tensors
            train_inp.iter_mut().zip(inp.iter()).for_each(|(a, b)| {
                a.0.borrow_mut().set_value(b.0.borrow().value().to_vec());
                a.1.borrow_mut().set_value(b.1.borrow().value().to_vec());
            })
        } else {
            panic!("Training must be defined before loading input values.");
        }
    }

    pub fn train(
        &mut self,
        inp: &[(TensorShared, TensorShared)],
        learning_rate: FloatType,
    ) -> TensorShared {
        self.load_correlations(inp);
        let calc = &self._train.as_ref().unwrap().1;
        let loss = calc.evaluate();
        calc.back_propagation();
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
        let param = self._layers.iter().flat_map(|l| l.parameters());
        if let Some(embed) = &self._embed {
            Box::new(embed.parameters().chain(param))
        }else{
            Box::new(param)
        }
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
            )),
            Box::new(LinearLayer::from_tensors(
                TensorShared::from_vector(vec![1., 1., 1., 1., 1., 1., 1., 1., 1.], (3, 3, 1)),
                None,
            )),
            Box::new(LinearLayer::from_tensors(
                TensorShared::from_vector(vec![1., 1., 1., 1., 1., 1.], (2, 3, 1)),
                None,
            )),
        ];
        let inp = TensorShared::from_vector(vec![1., 2.], (2, 1, 1));
        let mlp = MultiLayer::new_predictor(inp.shape(), None, (2, 1, 1), layers);
        let output = mlp.forward(&inp);
        assert_eq!(output.value_as_col_vector().unwrap(), vec![27., 27.]);
    }

    #[test]
    fn mlp_function_layer() {
        fn hej(t: &TensorShared) -> TensorShared {
            t.powf(1.)
        }
        let layers: Vec<Box<dyn Layer>> = vec![Box::new(FunctionLayer::new(&hej, "hej"))];
        let inp = TensorShared::from_vector(vec![1., 2.], (2, 1, 1));
        let mlp = MultiLayer::new_predictor(inp.shape(), None, (2, 1, 1), layers);
        let output = mlp.forward(&inp);
        assert_eq!(output.value_as_col_vector().unwrap(), vec![1., 2.]);
    }

    #[test]
    fn mlp_sigmoid_layer() {
        let layers: Vec<Box<dyn Layer>> = vec![Box::new(FunctionLayer::new(
            &FunctionLayer::sigmoid,
            "sigmoid",
        ))];
        let inp = TensorShared::from_vector(vec![1., 2.], (2, 1, 1));
        let mlp = MultiLayer::new_predictor(inp.shape(), None, (2, 1, 1), layers);
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
        ))];
        let inp = TensorShared::from_vector(vec![1., 2.], (2, 1, 1));
        let mlp = MultiLayer::new_predictor(inp.shape(), None, (2, 1, 1), layers);
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
