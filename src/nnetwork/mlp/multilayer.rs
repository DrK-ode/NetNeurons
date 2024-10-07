use std::{fmt::Display, time::Instant};

use rand::Rng;

use crate::nnetwork::{calc_node::FloatType, CalcNodeShared, Layer, Parameters};

use super::parameters::ParameterBundle;

pub type LossFuncType = dyn Fn(&CalcNodeShared, &CalcNodeShared) -> CalcNodeShared;

pub struct MultiLayer {
    _layers: Vec<Box<dyn Layer>>,
    _regularization: Option<FloatType>,
    _loss_func: Box<LossFuncType>,
}

impl MultiLayer {
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Self {
        MultiLayer { _layers: layers, _regularization: None, _loss_func: Box::new(&Self::neg_log_likelihood) }
    }
    
    pub fn set_loss_function(&mut self, f: &'static LossFuncType) {
        self._loss_func = Box::new(f);
    }
    
    pub fn set_regularization(&mut self, reg: Option<FloatType>){
        self._regularization = reg;
    }
   
    #[allow(clippy::borrowed_box)]
    pub fn get_layer(&self, i: usize) -> &Box<dyn Layer> {
        &self._layers[i]
    }

    pub fn forward(&self, inp: &CalcNodeShared) -> CalcNodeShared {
        self._layers
            .iter()
            .fold(inp.clone(), |out, layer| layer.forward(&out))
    }

    pub fn predict(&self, inp: &CalcNodeShared) -> CalcNodeShared {
        Self::collapse(&self.forward(inp))
    }

    fn collapse(inp: &CalcNodeShared) -> CalcNodeShared {
        let mut vec = vec![0.; inp.len()];
        let mut rnd = rand::thread_rng().gen_range(0. ..inp.borrow().vals().iter().sum());
        for (i, &v) in inp.borrow().vals().iter().enumerate() {
            rnd -= v;
            if rnd <= 0. || i + 1 == inp.len() {
                // Safe-guard against float precision errors
                vec[i] = 1.;
                break;
            }
        }
        CalcNodeShared::filled_from_shape(inp.shape(), vec)
    }

    fn calc_loss(
        &self,
        out: &[(CalcNodeShared, CalcNodeShared)],
    ) -> CalcNodeShared {
        let timer = Instant::now();
        let mut loss = out
            .iter()
            .map(|(out, truth)| (self._loss_func)(out, truth))
            .sum::<CalcNodeShared>()
            * CalcNodeShared::new_scalar(1. / out.len() as FloatType);

        if let Some(regularization) = self._regularization {
            if regularization <= 0. {
                panic!("Regularization coefficient must be positive.");
            }
            let regularization = CalcNodeShared::new_scalar(regularization);
            let n_param = self.param_iter().count();
            let n_param = CalcNodeShared::new_scalar(n_param as FloatType);
            // Mean of the sum of the squares of all parameters
            let param = self.param_iter();
            let reg_loss = param
                .map(|p| p.pow(&CalcNodeShared::new_scalar(2.)).sum())
                .sum::<CalcNodeShared>()
                * regularization
                / n_param;
            loss = loss + reg_loss;
        };
        println!("Calculating loss took {} µs", timer.elapsed().as_micros());
        loss
    }

    pub fn least_squares(inp: &CalcNodeShared, truth: &CalcNodeShared) -> CalcNodeShared {
        (inp - truth).pow(&CalcNodeShared::new_scalar(2.)).sum()
    }

    // Assumes the input can be treated as a probability distribution and that the truth is a one-hot vector
    pub fn neg_log_likelihood(inp: &CalcNodeShared, truth: &CalcNodeShared) -> CalcNodeShared {
        -(inp.element_wise_mul(truth)).sum().log()
    }
    
    pub fn loss(&self, 
        inp: &[(CalcNodeShared, CalcNodeShared)] ) -> CalcNodeShared {
            let out: Vec<_> = inp
                .iter()
                .map(|(inp, truth)| (self.forward(inp), truth.clone()))
                .collect();
            self.calc_loss(&out)
        }

    pub fn train(
        &mut self,
        inp: &[(CalcNodeShared, CalcNodeShared)],
        learning_rate: FloatType,
    ) -> CalcNodeShared {
        let loss = self.loss(inp);
        loss.back_propagation();
        self.decend_grad(learning_rate);

        loss
    }

    fn decend_grad(&self, learning_rate: FloatType) {
        self.param_iter().for_each(|p| p.decend_grad(learning_rate));
    }

    pub fn load_parameter_bundle(&self, bundle: &ParameterBundle) {
        bundle.load_parameters(&self._layers)
    }

    pub fn get_parameter_bundle(&self) -> ParameterBundle {
        ParameterBundle::new(&self._layers)
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
    fn param_iter(&self) -> Box<dyn Iterator<Item = &CalcNodeShared> + '_> {
        Box::new(self._layers.iter().flat_map(|l| l.param_iter()))
    }
}
