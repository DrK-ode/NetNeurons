use plotters::{
    chart::{ChartBuilder, LabelAreaPosition},
    prelude::{BitMapBackend, Cross, IntoDrawingArea, Rectangle},
    style::{RGBAColor, ShapeStyle, RED, WHITE},
};
use rand_distr::{Distribution, Uniform};
use std::{error::Error, fmt::Display, ops::Range, time::Instant};

use crate::nnetwork::{
    loss_functions::least_squares, CalcNode, FloatType, FunctionLayer, Layer, LinearLayer,
    MultiLayer, Parameters,
};

/// Manages the construction and training of a network that decides what color a pixel should have.
pub struct ReColor<T>
where
    T: Fn((FloatType, FloatType)) -> [bool; 3],
{
    _color_key: T,
    _mlp: MultiLayer,
    _regularization: Option<FloatType>,
    _training_results: Vec<(FloatType,FloatType)>,
}

impl<T> ReColor<T>
where
    T: Fn((FloatType, FloatType)) -> [bool; 3],
{
    // Helps ctor
    fn create_layers(n_hidden_layers: usize, layer_size: usize) -> Vec<Box<dyn Layer>> {
        const BIASED_LAYERS: bool = true;
        const INPUT_DIM: usize = 2;
        const OUTPUT_DIM: usize = 3;
        let non_linearity =
            FunctionLayer::new(&FunctionLayer::sigmoid, "Sigmoid", "Non-linearity layer");
        // ReLU has major problems with convergence and a tendancy till zero out the whole network with the scheme used here.
        //let non_linearity = FunctionLayer::new(&FunctionLayer::leaky_relu, "Leaky ReLU", "Non-linearity layer");
        let mut layers: Vec<Box<dyn Layer>> = Vec::new();

        // Input layer
        layers.push(Box::new(LinearLayer::new_rand(
            layer_size,
            INPUT_DIM,
            BIASED_LAYERS,
            "Resizing layer (in)",
        )));
        layers.push(Box::new(non_linearity.clone()));

        // Hidden layers
        for n in 0..n_hidden_layers {
            layers.push(Box::new(LinearLayer::new_rand(
                layer_size,
                layer_size,
                BIASED_LAYERS,
                &format!("Hidden layer {n}"),
            )));
            layers.push(Box::new(non_linearity.clone()));
        }

        // Output layer
        layers.push(Box::new(LinearLayer::new_rand(
            OUTPUT_DIM,
            layer_size,
            BIASED_LAYERS,
            "Resizing layer (out)",
        )));
        layers.push(Box::new(non_linearity.clone()));

        layers
    }

    /// After each linear layer a non-linear [FunctionLayer] is inserted.
    pub fn new(
        color_key: T,
        n_hidden_layers: usize,
        layer_size: usize,
        regularization: Option<FloatType>,
    ) -> ReColor<T> {
        let mut mlp = MultiLayer::new(Self::create_layers(n_hidden_layers, layer_size));
        mlp.set_regularization(regularization);
        mlp.set_loss_function(&least_squares);
        ReColor {
            _color_key: color_key,
            _mlp: mlp,
            _regularization: regularization,
            _training_results: Vec::new(),
        }
    }

    fn coords_to_rgb(&self, coords: (FloatType, FloatType)) -> CalcNode {
        CalcNode::new_col_vector(
            (self._color_key)(coords)
                .into_iter()
                .map(|boolean| if boolean { 1. } else { 0. })
                .collect(),
        )
    }

    /// Returns predicted RGB values for the specified coordinates.
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

    // Creates a list of tuples containing input coords and the correct color
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
                let color = self.coords_to_rgb(coords);
                let coords = CalcNode::new_col_vector(vec![coords.0, coords.1]);
                (coords, color)
            })
            .collect()
    }

    /// Trains the network for the specified number of cycles. Each cycles uses ´batch_size´ data points.
    /// The learning rate is a range from highest to lowest which will be logspaced so that the learning rate get lower for each cycle.
    ///
    /// Returns a vector of learning rates and loss values
    pub fn train(
        &mut self,
        cycles: usize,
        batch_size: usize,
        learning_rate: Range<FloatType>,
        x_range: &Range<FloatType>,
        y_range: &Range<FloatType>,
        verbose: bool,
    ) -> &[(FloatType, FloatType)] {
        let timer = Instant::now();
        self._training_results.clear();
        let mut loss = 0.;
        let learning_rate_log_step = if cycles < 2 {
            learning_rate.start
        } else {
            (learning_rate.end.ln() - learning_rate.start.ln()) / (cycles - 1) as FloatType
        };
        for n in 0..cycles {
            let correlations = self.calc_correlations(batch_size, x_range, y_range);
            let timer = Instant::now();
            let learning_rate =
                (learning_rate.start.ln() + learning_rate_log_step * n as FloatType).exp();
            loss = self._mlp.train(&correlations, learning_rate);

            self._training_results.push((learning_rate, loss));

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
        &self._training_results
    }

    pub fn export_parameters(&self, filename: &str) -> std::io::Result<String> {
        self._mlp.export_parameters(filename)
    }

    pub fn import_parameters(&mut self, filename: &str) -> Result<(), std::io::Error> {
        self._mlp.import_parameters(filename)
    }

    /// Plots the colours predicted by the network for a sample of coordinates.
    pub fn plot_predictions(
        &self,
        x_range: &Range<FloatType>,
        y_range: &Range<FloatType>,
        x_divisions: u32,
        filename: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let size = (x_range.end - x_range.start, y_range.end - y_range.start);
        let division = (
            x_divisions,
            (x_divisions as FloatType * size.1 / size.0) as usize,
        );
        let step = (
            size.0 / division.0 as FloatType,
            size.1 / division.1 as FloatType,
        );

        const X_PIXELS: u32 = 1000;
        let drawing_area = BitMapBackend::new(
            filename,
            (X_PIXELS, (X_PIXELS as FloatType * size.1 / size.0) as u32),
        )
        .into_drawing_area();
        drawing_area.fill(&WHITE)?;
        let mut chart = ChartBuilder::on(&drawing_area)
            .x_label_area_size(0)
            .y_label_area_size(0)
            .set_label_area_size(LabelAreaPosition::Left, 40)
            .set_label_area_size(LabelAreaPosition::Bottom, 40)
            .build_cartesian_2d(x_range.start..x_range.end, y_range.start..y_range.end)?;

        chart.draw_series((0..division.0).flat_map(|xi| {
            let xl = x_range.start + step.0 * xi as FloatType;
            let xr = xl + step.0;
            let xm = xl + step.0 * 0.5;
            (0..division.1).map(move |yi| {
                let yt = y_range.start + step.1 * yi as FloatType;
                let yb = yt + step.1;
                let ym = yt + step.1 * 0.5;
                let rgb: Vec<_> = self
                    .predict((xm, ym))
                    .iter()
                    .map(|c| (c * 255.0) as u8)
                    .collect();
                let color = RGBAColor(rgb[0], rgb[1], rgb[2], 1.);
                Rectangle::new(
                    [(xl, yt), (xr, yb)],
                    ShapeStyle {
                        color,
                        filled: true,
                        stroke_width: 0,
                    },
                )
            })
        }))?;
        chart.configure_mesh().disable_mesh().draw()?;

        drawing_area.present()?;
        Ok(())
    }
    
    /// Plots a diagram of log(loss) vs p(learning rate).
    pub fn plot_training_progress(
        &self,
        filename: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if self._training_results.is_empty(){
            return Err(Box::new(TrainingError{}));
        }
        const X_PIXELS: u32 = 1024;
        const Y_PIXELS: u32 = 768;
        let drawing_area = BitMapBackend::new(filename, (X_PIXELS, Y_PIXELS)).into_drawing_area();
        let min = self._training_results
            .iter()
            .fold((FloatType::MAX, FloatType::MAX), |acc, (x, y)| {
                (x.min(acc.0), y.min(acc.1))
            });
        let max = self._training_results
            .iter()
            .fold((FloatType::MIN, FloatType::MIN), |acc, (x, y)| {
                (x.max(acc.0), y.max(acc.1))
            });
        let x_begin = -max.0.log10();
        let x_end = -min.0.log10();
        let y_begin = min.1.log10();
        let y_end = max.1.log10();
        drawing_area.fill(&WHITE)?;
        let mut chart = ChartBuilder::on(&drawing_area)
            .x_label_area_size(0)
            .y_label_area_size(0)
            .set_label_area_size(LabelAreaPosition::Left, 40)
            .set_label_area_size(LabelAreaPosition::Bottom, 40)
            .build_cartesian_2d(x_begin..x_end, y_begin..y_end)?;

        chart.draw_series(self._training_results.iter().map(|(learning_rate, loss)| {
            Cross::new((-learning_rate.log10(), loss.log10()), 5, RED)
        }))?;
        chart
            .configure_mesh()
            .x_desc("Neg log10 learning rate")
            .y_desc("Log10 Loss")
            .draw()?;

        drawing_area.present()?;
        Ok(())
    }
}

#[derive(Debug)]
struct TrainingError{}
impl Display for TrainingError{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f,"No training data available.")
    }
}
impl Error for TrainingError{}
