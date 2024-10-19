use std::{cmp::Ordering, ops::Range, time::Instant};
use rand_distr::{Distribution, Uniform};

use crate::nnetwork::{loss_functions::least_squares, CalcNode, FloatType, FunctionLayer, Layer, LinearLayer, MultiLayer, Parameters};

use super::{color_key::{Color, ColorFunction}, ColorKey};

pub struct ColorPredictor {
    _color_key: ColorKey,
    _mlp: MultiLayer,
    _regularization: Option<FloatType>,
}

impl ColorPredictor {
    fn create_layers(n_hidden_layers: usize, layer_size: usize) -> Vec<Box<dyn Layer>> {
        const BIASED_LAYERS: bool = true;
        const INPUT_DIM: usize = 2;
        const OUTPUT_DIM: usize = 4;
        let non_linearity = FunctionLayer::new(&FunctionLayer::tanh, "Tanh", "Non-linearity layer");
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

        layers.push(Box::new(FunctionLayer::new(
            &FunctionLayer::softmax,
            "SoftMax",
            "Probability producing layer",
        )));
        layers
    }

    pub fn new(
        color_func: ColorFunction,
        n_hidden_layers: usize,
        layer_size: usize,
        regularization: Option<FloatType>,
    ) -> ColorPredictor {
        let mut mlp = MultiLayer::new(Self::create_layers(n_hidden_layers, layer_size));
        mlp.set_regularization(regularization);
        mlp.set_loss_function(&least_squares);
        ColorPredictor {
            _color_key: ColorKey::new(color_func),
            _mlp: mlp,
            _regularization: regularization,
        }
    }

    pub fn predict(&self, coords: (FloatType, FloatType)) -> Color {
        let coords = CalcNode::new_col_vector(vec![coords.0, coords.1]);
        let (max_index,_max_value) = self
            ._mlp
            .forward(&coords)
            .copy_vals()
            .iter()
            .enumerate()
            .max_by(|(_ai, &av), (_bi, &bv)| {
                if av > bv {
                    Ordering::Greater
                } else if av < bv {
                    Ordering::Less
                } else {
                    Ordering::Equal
                }
            }).unwrap();
        Color::from(max_index)
    }
    
    fn calc_correlations(&self, batch_size:usize, x_range: &Range<FloatType>, y_range: &Range<FloatType>) -> Vec<(CalcNode,CalcNode)>{
        let mut rng = rand::thread_rng();
        let x_dist = Uniform::from(x_range.clone());
        let y_dist = Uniform::from(y_range.clone());
        (0..batch_size).map(|_| {
            let coords = (x_dist.sample(&mut rng), y_dist.sample(&mut rng));
            let mut color = CalcNode::filled_from_shape((1,4),vec![0.;4]);
            color.set_value_indexed(self._color_key.color(coords).into(),1.);
            (CalcNode::new_col_vector(vec![coords.0,coords.1]),color)
        }).collect()
    }
    
    pub fn train(&mut self, cycles:usize,batch_size:usize,learning_rate: FloatType, x_range: &Range<FloatType>, y_range: &Range<FloatType>, verbose: bool) -> FloatType {
        let timer = Instant::now();
        let mut loss = 0.;
        for n in 0..cycles {
            let correlations = self.calc_correlations(batch_size,x_range,y_range);
            let timer = Instant::now();
            loss = self._mlp.train(&correlations, learning_rate);

            // Provide some per cycle stats
            if verbose {
                let width = (cycles as f64).log10() as usize + 1;
                println!(
                    "Cycle #{n: >width$}: [ loss: {:.3e}, duration: {} µs ]",
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
        loss
    }
}
