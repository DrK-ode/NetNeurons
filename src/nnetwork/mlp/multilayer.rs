use std::{fmt::Display, time::Instant};

use rand::Rng;

use crate::nnetwork::{calc_node::FloatType, CalcNode, Layer, Parameters};

use super::{loss_functions::neg_log_likelihood, parameter_bundle::ParameterBundle};

pub type LossFuncType = dyn Fn(&CalcNode, &CalcNode) -> CalcNode;

pub struct MultiLayer {
    _layers: Vec<Box<dyn Layer>>,
    _regularization: Option<FloatType>,
    _loss_func: Box<LossFuncType>,
}

impl MultiLayer {
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Self {
        MultiLayer {
            _layers: layers,
            _regularization: None,
            _loss_func: Box::new(&neg_log_likelihood),
        }
    }

    pub fn set_loss_function(&mut self, f: &'static LossFuncType) {
        self._loss_func = Box::new(f);
    }

    pub fn set_regularization(&mut self, reg: Option<FloatType>) {
        self._regularization = reg;
    }

    pub fn get_layer(&self, i: usize) -> &dyn Layer {
        self._layers[i].as_ref()
    }

    pub fn forward(&self, inp: &CalcNode) -> CalcNode {
        self._layers
            .iter()
            .fold(inp.clone(), |out, layer| layer.forward(&out))
    }

    pub fn predict(&self, inp: &CalcNode) -> CalcNode {
        Self::collapse(&self.forward(inp))
    }

    fn collapse(inp: &CalcNode) -> CalcNode {
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
        CalcNode::filled_from_shape(inp.shape(), vec)
    }

    fn calc_regularization(&self) -> CalcNode {
        if let Some(regularization) = self._regularization {
            if regularization <= 0. {
                panic!("Regularization coefficient must be positive.");
            }
            let regularization = CalcNode::new_scalar(regularization);
            let n_param = self.param_iter().count();
            let n_param = CalcNode::new_scalar(n_param as FloatType);
            // Mean of the sum of the squares of all parameters
            let param = self.param_iter();
            param
                .map(|p| p.pow(&CalcNode::new_scalar(2.)).sum())
                .sum::<CalcNode>()
                * regularization
                / n_param
        } else {
            CalcNode::new_scalar(0.)
        }
    }

    pub fn loss(&self, inp: &[(CalcNode, CalcNode)]) -> CalcNode {
        let timer = Instant::now();
        let loss = inp
            .iter()
            .map(|(inp, truth)| (self._loss_func)(&self.forward(inp), truth))
            .sum::<CalcNode>()
            * CalcNode::new_scalar(1. / inp.len() as FloatType);
        let reg = self.calc_regularization();
        println!("Calculating loss took {} µs", timer.elapsed().as_micros());
        loss + reg
    }

    pub fn train(&mut self, inp: &[(CalcNode, CalcNode)], learning_rate: FloatType) -> FloatType {
        let mut loss = self.loss(inp);
        loss.back_propagation();
        self.decend_grad(learning_rate);

        loss.value_indexed(0)
    }

    fn decend_grad(&mut self, learning_rate: FloatType) {
        self.param_iter_mut()
            .for_each(|p| p.decend_grad(learning_rate));
    }

    pub fn load_parameter_bundle(&mut self, bundle: &ParameterBundle) {
        bundle.load_parameters_into(&mut self._layers)
    }

    pub fn get_parameter_bundle(&self) -> ParameterBundle {
        ParameterBundle::new_from_layers(&self._layers)
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
    fn param_iter(&self) -> Box<dyn Iterator<Item = &CalcNode> + '_> {
        Box::new(self._layers.iter().flat_map(|l| l.param_iter()))
    }
    fn param_iter_mut(&mut self) -> Box<dyn Iterator<Item = &mut CalcNode> + '_> {
        Box::new(self._layers.iter_mut().flat_map(|l| l.param_iter_mut()))
    }
}
