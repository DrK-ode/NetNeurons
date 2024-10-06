use rand::Rng;
use std::time::Instant;

use crate::{
    data_preparing::data_set::{DataSet, DataSetError},
    nnetwork::{FunctionLayer, LinearLayer, Parameters, ReshapeLayer},
};

use super::{CalcNodeShared, FloatType, Layer, MultiLayer, ParameterBundle};

pub struct ReText {
    _dataset: DataSet,
    _mlp: MultiLayer,
    _block_size: usize,
    _embedding: bool,
}

impl ReText {
    pub fn create_layers(
        n_chars: usize,
        block_size: usize,
        embed_dim: Option<usize>,
        n_hidden_layers: usize,
        layer_dim: usize,
    ) -> Vec<Box<dyn Layer>> {
        let mut layers: Vec<Box<dyn Layer>> = Vec::new();
        let non_linearity = FunctionLayer::new(&FunctionLayer::tanh, "Tanh", "Non-linearity layer");
        const BIASED_LAYERS: bool = true;

        //Embed
        if let Some(embed_dim) = embed_dim {
            let embed_layer = LinearLayer::from_rand(embed_dim, n_chars, false, "Embedding layer");
            let reshape_layer = ReshapeLayer::new((block_size * embed_dim, 1), "Reshaping layer");
            let resize_layer = LinearLayer::from_rand(
                layer_dim,
                block_size * embed_dim,
                BIASED_LAYERS,
                "Resizing layer (in)",
            );
            layers.push(Box::new(embed_layer));
            layers.push(Box::new(reshape_layer));
            layers.push(Box::new(non_linearity.clone()));
            layers.push(Box::new(resize_layer));
        } else {
            let resize_layer =
                LinearLayer::from_rand(layer_dim, n_chars, BIASED_LAYERS, "Resizing layer (in)");
            layers.push(Box::new(resize_layer));
        }
        layers.push(Box::new(non_linearity.clone()));

        // Hidden layers
        for n in 0..n_hidden_layers {
            layers.push(Box::new(LinearLayer::from_rand(
                layer_dim,
                layer_dim,
                BIASED_LAYERS,
                &format!("Hidden layer {n}"),
            )));
            layers.push(Box::new(non_linearity.clone()));
        }

        // Deembed
        layers.push(Box::new(LinearLayer::from_rand(
            n_chars,
            layer_dim,
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
        data: DataSet,
        block_size: usize,
        embed_dim: Option<usize>,
        n_hidden_layers: usize,
        layer_dim: usize,
        regularization: Option<FloatType>,
    ) -> ReText {
        let n_chars = data.number_of_chars();
        let layers =
            Self::create_layers(n_chars, block_size, embed_dim, n_hidden_layers, layer_dim);
        let mut mlp = MultiLayer::new(layers);
        mlp.set_regularization(regularization);
        mlp.set_loss_function(&MultiLayer::neg_log_likelihood);
        ReText {
            _dataset: data,
            _block_size: block_size,
            _mlp: mlp,
            _embedding: embed_dim.is_some(),
        }
    }

    fn validate(&self, data_size: usize) -> FloatType {
        let data = self._dataset.training_data();
        let correlations = self.extract_correlations(data, data_size);
        self._mlp.loss(&correlations).value_indexed(0)
    }

    pub fn train(
        &mut self,
        cycles: usize,
        learning_rate: FloatType,
        data_size: usize,
        verbose: bool,
    ) {
        let timer = Instant::now();
        for n in 0..cycles {
            let data = self._dataset.training_data();
            let correlations = self.extract_correlations(data, data_size);
            let timer = Instant::now();
            let loss = self._mlp.train(&correlations, learning_rate);

            if verbose {
                let width = (cycles as f64).log10() as usize + 1;
                println!(
                    "Cycle #{n: >width$}: [ loss: {:.3e}, duration: {} µs ]",
                    loss.value_indexed(0),
                    timer.elapsed().as_micros()
                );
            }
        }
        println!(
            "Trained network with {} parameters for {cycles} cycles in {} ms.",
            self._mlp.param_iter().map(|p| p.len()).sum::<usize>(),
            timer.elapsed().as_millis()
        );

        let validation = self.validate(usize::MAX);
        println!("Validation loss: {}", validation);
    }

    pub fn embed(&self, inp: &str) -> CalcNodeShared {
        let inp = self._dataset.encode(inp).unwrap();
        if !self._embedding {
            return inp.clone();
        }
        self._mlp.get_layer(0).forward(&inp)
    }

    // Returns a list of all correlations in the data encoded as a tuple of Matrix(m*n) and ColumnVector(n).
    fn extract_correlations(
        &self,
        data: &[String],
        n: usize,
    ) -> Vec<(CalcNodeShared, CalcNodeShared)> {
        let n_lines = data.len();
        let mut correlations = Vec::new();
        let pad = "^".to_string().repeat(self._block_size);
        let start_idx = rand::thread_rng().gen_range(0..n_lines);
        let mut line_idx = start_idx;
        while correlations.len() < n {
            let line = &data[line_idx];
            let s = "".to_string() + &pad + line + "^";
            s.char_indices()
                .zip(s.char_indices().skip(self._block_size))
                .map(|((i, _prev), (j, next))| {
                    let prev = &s[i..j];
                    let next = next.to_string();
                    (
                        self._dataset
                            .encode(prev)
                            .expect("Cannot encode character: {prev}"),
                        self._dataset
                            .encode(&next)
                            .expect("Cannot encode character: {next}"),
                    )
                })
                .for_each(|corr| {
                    correlations.push(corr);
                });
            line_idx += 1;
            if line_idx >= data.len() {
                line_idx = 0;
            }
            if line_idx == start_idx {
                break;
            }
        }
        correlations.truncate(n);
        correlations
    }

    pub fn predict(
        &mut self,
        seed_string: &str,
        number_of_characters: usize,
    ) -> Result<String, DataSetError> {
        let pad = "^".to_string().repeat(self._block_size);
        let mut s = pad + seed_string;
        if s.len() < self._block_size {
            panic!(
                "Aborting, cannot extrapolate from string shorter than {}.",
                self._block_size
            );
        }
        let range = (s.len() - self._block_size)..;
        // The following line break upon non ascii input
        let mut last = self._dataset.encode(&s[range])?;
        for _ in 0..number_of_characters {
            last = self._mlp.predict(&last);
            let c = self._dataset.decode(&last)?;
            if c == '^' {
                break;
            }
            s.push(c);
        }
        Ok(s[self._block_size..].to_string())
    }

    pub fn characters(&self) -> &[char] {
        self._dataset.characters()
    }

    pub fn get_parameter_bundle(&self) -> ParameterBundle {
        self._mlp.get_parameter_bundle()
    }

    pub fn load_trainer_parameter_bundle(&mut self, bundle: &ParameterBundle) {
        self._mlp.load_parameter_bundle(bundle)
    }

    pub fn load_predictor_parameter_bundle(&mut self, bundle: &ParameterBundle) {
        self._mlp.load_parameter_bundle(bundle)
    }
}
