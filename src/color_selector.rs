use rand_distr::{Distribution, Uniform};
use std::{ops::Range, time::Instant};

use crate::nnetwork::{
    loss_functions::least_squares, CalcNode, FloatType, FunctionLayer, Layer, LinearLayer,
    MultiLayer, Parameters,
};

pub type ColorFunction = Box<dyn Fn((FloatType, FloatType)) -> [bool; 3]>;

pub struct ColorSelector {
    _color_key: ColorFunction,
    _mlp: MultiLayer,
    _regularization: Option<FloatType>,
}

impl ColorSelector {
    fn create_layers(n_hidden_layers: usize, layer_size: usize) -> Vec<Box<dyn Layer>> {
        const BIASED_LAYERS: bool = true;
        const INPUT_DIM: usize = 2;
        const OUTPUT_DIM: usize = 3;
        let non_linearity = FunctionLayer::new(&FunctionLayer::sigmoid, "Sigmoid", "Non-linearity layer");
        // ReLU has major problems with convergence and a tendancy till zero out the whole network with the scheme used here.
        //let non_linearity = FunctionLayer::new(&FunctionLayer::leaky_relu, "Leaky ReLU", "Non-linearity layer");
        let mut layers: Vec<Box<dyn Layer>> = Vec::new();

        layers.push(Box::new(LinearLayer::from_rand(
            layer_size,
            INPUT_DIM,
            BIASED_LAYERS,
            "Resizing layer (in)",
        )));
        layers.push(Box::new(non_linearity.clone()));

        // Hidden layers
        for n in 0..n_hidden_layers {
            layers.push(Box::new(LinearLayer::from_rand(
                layer_size,
                layer_size,
                BIASED_LAYERS,
                &format!("Hidden layer {n}"),
            )));
            layers.push(Box::new(non_linearity.clone()));
        }
        layers.push(Box::new(LinearLayer::from_rand(
            OUTPUT_DIM,
            layer_size,
            BIASED_LAYERS,
            "Resizing layer (out)",
        )));
        layers.push(Box::new(non_linearity.clone()));

        layers
    }

    pub fn new(
        color_func: ColorFunction,
        n_hidden_layers: usize,
        layer_size: usize,
        regularization: Option<FloatType>,
    ) -> ColorSelector {
        let mut mlp = MultiLayer::new(Self::create_layers(n_hidden_layers, layer_size));
        mlp.set_regularization(regularization);
        mlp.set_loss_function(&least_squares);
        ColorSelector {
            _color_key: color_func,
            _mlp: mlp,
            _regularization: regularization,
        }
    }

    fn color(&self, coords: (FloatType, FloatType)) -> CalcNode {
        CalcNode::new_col_vector(
            (self._color_key)(coords)
                .into_iter()
                .map(|boolean| if boolean { 1. } else { 0. })
                .collect(),
        )
    }

    pub fn predict(&self, coords: (FloatType, FloatType)) -> [FloatType; 3] {
        let coords = CalcNode::new_col_vector(vec![coords.0, coords.1]);
        self._mlp
            .forward(&coords)
            .copy_vals()
            .try_into()
            .unwrap_or_else(|vec: Vec<FloatType>| {
                panic!("Expected a Vec of length {} but it was {}", 3, vec.len())
            })
    }

    fn calc_correlations(
        &self,
        batch_size: usize,
        x_range: &Range<FloatType>,
        y_range: &Range<FloatType>,
    ) -> Vec<(CalcNode, CalcNode)> {
        let mut rng = rand::thread_rng();
        let x_dist = Uniform::from(x_range.clone());
        let y_dist = Uniform::from(y_range.clone());
        (0..batch_size)
            .map(|_| {
                let coords = (x_dist.sample(&mut rng), y_dist.sample(&mut rng));
                let color = self.color(coords);
                let coords = CalcNode::new_col_vector(vec![coords.0, coords.1]);
                (coords, color)
            })
            .collect()
    }

    pub fn train(
        &mut self,
        cycles: usize,
        batch_size: usize,
        learning_rate: Range<FloatType>,
        x_range: &Range<FloatType>,
        y_range: &Range<FloatType>,
        verbose: bool,
    ) -> Vec<(FloatType, FloatType)> {
        let timer = Instant::now();
        let mut training_points = Vec::new();
        let mut loss = 0.;
        let learning_rate_log_step =
            (learning_rate.end.ln() - learning_rate.start.ln()) / (cycles - 1) as FloatType;
        for n in 0..cycles {
            let correlations = self.calc_correlations(batch_size, x_range, y_range);
            let timer = Instant::now();
            let learning_rate =
                (learning_rate.start.ln() + learning_rate_log_step * n as FloatType).exp();
            loss = self._mlp.train(&correlations, learning_rate);

            training_points.push((learning_rate, loss));

            // Provide some per cycle stats
            if verbose {
                let width = (cycles as f64).log10() as usize + 1;
                println!(
                    "Cycle #{n: >width$}, learning_rate: {learning_rate:.2e} [ loss: {:.3e}, duration: {} µs ]",
                    loss,
                    timer.elapsed().as_micros()
                );
            }
        }
        println!(
            "Trained network with {} parameters for {cycles} cycles in {} ms achieving a loss of: {:.3e}",
            self._mlp.param_iter().map(|p| p.len()).sum::<usize>(),
            timer.elapsed().as_millis(), loss
        );
        training_points
    }

    pub fn export_parameters(&self, filename: &str) -> std::io::Result<String> {
        self._mlp.export_parameters(filename)
    }

    pub fn import_parameters(&mut self, filename: &str) -> Result<(), std::io::Error> {
        self._mlp.import_parameters(filename)
    }
}
