use std::{fmt::Display, time::Instant};

use crate::nnetwork::{
    calculation_nodes::{FloatType, NetworkCalculation, TensorShared}, mlp::ParameterBundle, Layer, Parameters, TensorShape
};

type LossFuncType = dyn Fn(&TensorShared, &TensorShared)->TensorShared;
 
pub struct Trainer {
    _layers: Vec<Box<dyn Layer>>,
    _inp: Vec<(TensorShared, TensorShared)>,
    _calc: NetworkCalculation,
    _loss_function: Box<LossFuncType>
}

impl Trainer {
    pub fn new(
        inp_shape: TensorShape,
        out_shape: TensorShape,
        batch_size: usize,
        layers: Vec<Box<dyn Layer>>,
        regularization: Option<FloatType>,
        loss_func: &'static LossFuncType
    ) -> Self {
        let inp = (0..batch_size)
            .map(|_| {
                (
                    TensorShared::from_shape((inp_shape.0, inp_shape.1, 1)),
                    TensorShared::from_shape((out_shape.0, out_shape.1, 1)),
                )
            })
            .collect::<Vec<_>>();
        let calc = Self::define_train_calc(&layers, &inp, regularization, loss_func);
        Trainer {
            _layers: layers,
            _inp: inp,
            _calc: calc,
            _loss_function: Box::new(loss_func)
        }
    }
    
    pub fn loss(&self, inp: &TensorShared, truth: &TensorShared) -> TensorShared{
        (self._loss_function)(inp,truth)
    }

    fn define_train_calc(
        layers: &[Box<dyn Layer>],
        inp: &[(TensorShared, TensorShared)],
        regularization: Option<FloatType>,
        loss_func: &'static LossFuncType,
    ) -> NetworkCalculation {
        // Copy input data into place
        let outs = inp
            .iter()
            .map(|(inp, truth)| {
                (
                    layers.iter().fold(inp.clone(), |out, l| l.forward(&out)),
                    truth.clone(),
                )
            })
            .collect::<Vec<_>>();

        let timer = Instant::now();
        let mut loss = outs
            .iter()
            .map(|(out, truth)| (loss_func)(out, truth))
            .sum::<TensorShared>()
            * TensorShared::from_scalar(1. / outs.len() as FloatType);

        if let Some(regularization) = regularization {
            if regularization <= 0. {
                panic!("Regularization coefficient must be positive.");
            }
            let regularization = TensorShared::from_scalar(regularization);
            let n_param = Self::parameters_from_layers(layers).count();
            let n_param = TensorShared::from_scalar(n_param as FloatType);
            // Mean of the sum of the squares of all parameters
            let param = Self::parameters_from_layers(layers);
            let reg_loss =
                param.map(|p| p.powf(2.).sum()).sum::<TensorShared>() * regularization / n_param;
            loss = loss + reg_loss;
        };
        println!(
            "Defining calculation took {} µs",
            timer.elapsed().as_micros()
        );

        let timer = Instant::now();
        let calc = NetworkCalculation::new(&loss);
        println!(
            "Topological sorting took {} µs",
            timer.elapsed().as_micros()
        );
        calc
    }

    pub fn least_squares(inp: &TensorShared, truth: &TensorShared) -> TensorShared {
        (inp - truth).powf(2.).sum()
    }

    // Assumes the input can be treated as a probability distribution and that the truth is a one-hot vector
    pub fn neg_log_likelihood(inp: &TensorShared, truth: &TensorShared) -> TensorShared {
        -(inp * truth).sum().log()
    }

    pub fn train(
        &mut self,
        inp: &[(TensorShared, TensorShared)],
        learning_rate: FloatType,
    ) -> TensorShared {
        // Copy values over to the input tensors
        self._inp.iter_mut().zip(inp.iter()).for_each(|(a, b)| {
            a.0.borrow_mut().set_value(b.0.borrow().value().to_vec());
            a.1.borrow_mut().set_value(b.1.borrow().value().to_vec());
        });
        let loss = self._calc.evaluate();
        self._calc.back_propagation();
        self.decend_grad(learning_rate);
        loss
    }

    fn decend_grad(&self, learning_rate: FloatType) {
        self.param_iter().for_each(|p| p.decend_grad(learning_rate));
    }
    
    pub fn load_parameter_bundle(&self, bundle: &ParameterBundle){
        bundle.load_parameters(&self._layers)
    }
    
    pub fn get_parameter_bundle(&self) -> ParameterBundle {
        ParameterBundle::new(&self._layers)
    }
}

impl Display for Trainer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "MLP: [")?;
        for layer in &self._layers {
            layer.fmt(f)?;
        }
        writeln!(f, "]")
    }
}

impl Trainer {
    fn parameters_from_layers(
        layers: &[Box<dyn Layer>],
    ) -> Box<dyn Iterator<Item = &TensorShared> + '_> {
        Box::new(layers.iter().flat_map(|l| l.param_iter()))
    }
}
impl Parameters for Trainer {
    fn param_iter(&self) -> Box<dyn Iterator<Item = &TensorShared> + '_> {
        Self::parameters_from_layers(&self._layers)
    }
}
