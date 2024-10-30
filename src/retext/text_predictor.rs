use rand::Rng;
use std::io::Error;
use std::time::Instant;

use crate::{
    nnetwork::{FunctionLayer, LinearLayer, Parameters, ReshapeLayer},
    retext::char_set::{CharSet, DataSetError},
};

use crate::nnetwork::{loss_functions::neg_log_likelihood, CalcNode, FloatType, Layer, MultiLayer};

const SENTINEL_TOKEN: &str = "^";

pub struct ReText {
    _dataset: CharSet,
    _mlp: MultiLayer,
    _block_size: usize,
}

impl ReText {
    fn create_layers(
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
            let embed_layer = LinearLayer::new_rand(embed_dim, n_chars, false, "Embedding layer");
            let reshape_layer = ReshapeLayer::new((block_size * embed_dim, 1), "Reshaping layer");
            let resize_layer = LinearLayer::new_rand(
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
                LinearLayer::new_rand(layer_dim, n_chars, BIASED_LAYERS, "Resizing layer (in)");
            layers.push(Box::new(resize_layer));
        }
        layers.push(Box::new(non_linearity.clone()));

        // Hidden layers
        for n in 0..n_hidden_layers {
            layers.push(Box::new(LinearLayer::new_rand(
                layer_dim,
                layer_dim,
                BIASED_LAYERS,
                &format!("Hidden layer {n}"),
            )));
            layers.push(Box::new(non_linearity.clone()));
        }

        // Deembed
        layers.push(Box::new(LinearLayer::new_rand(
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
        mut data: CharSet,
        block_size: usize,
        embed_dim: Option<usize>,
        n_hidden_layers: usize,
        layer_dim: usize,
        regularization: Option<FloatType>,
    ) -> ReText {
        data.add_character(SENTINEL_TOKEN.chars().nth(0).unwrap());
        let n_chars = data.number_of_chars();
        let layers =
            Self::create_layers(n_chars, block_size, embed_dim, n_hidden_layers, layer_dim);
        let mut mlp = MultiLayer::new(layers);
        mlp.set_regularization(regularization);
        mlp.set_loss_function(&neg_log_likelihood);
        ReText {
            _dataset: data,
            _block_size: block_size,
            _mlp: mlp,
        }
    }

    fn validate(&self, data_size: usize) -> FloatType {
        let data = self._dataset.validation_data();
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
        let mut loss = 0.;
        for n in 0..cycles {
            let data = self._dataset.training_data();
            let correlations = self.extract_correlations(data, data_size);
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

        let validation = self.validate(data_size);
        println!("Validation loss: {}", validation);
    }

    fn get_all_correlations_from_str(&self, line: &str) -> Vec<(CalcNode, CalcNode)> {
        // Pad the string with the sentinel token
        let pad = SENTINEL_TOKEN.to_string().repeat(self._block_size);
        let s = pad + line + SENTINEL_TOKEN;
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
            .collect()
    }

    // Returns a list of all correlations in the data encoded as a tuple of Matrix(m*n) and ColumnVector(n).
    fn extract_correlations(&self, data: &[String], n: usize) -> Vec<(CalcNode, CalcNode)> {
        let n_lines = data.len();
        let mut correlations = Vec::new();
        let start_idx = rand::thread_rng().gen_range(0..n_lines);
        let mut line_idx = start_idx;
        while correlations.len() < n {
            let line = &data[line_idx];
            correlations.append(&mut self.get_all_correlations_from_str(line));
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
        assert!(
            !seed_string.is_empty(),
            "Cannot extrapolate from empty string."
        );
        // Pad the string with the sentinel token
        let mut str = SENTINEL_TOKEN.to_string().repeat(self._block_size - 1) + seed_string;
        for _ in 0..number_of_characters {
            // The following line break upon non ascii input
            let mut last = self._dataset.encode(&str[str.len() - self._block_size..])?;
            last = self._mlp.forward(&last).collapse();
            let c = self._dataset.decode_char(&last)?;
            if c == SENTINEL_TOKEN.chars().nth(0).unwrap() {
                break;
            }
            str.push(c);
        }
        Ok(str[self._block_size - 1..].to_string())
    }

    pub fn characters(&self) -> &[char] {
        self._dataset.characters()
    }

    pub fn export_parameters(&self, filename: &str) -> std::io::Result<String> {
        self._mlp.export_parameters(filename)
    }

    pub fn import_parameters(&mut self, filename: &str) -> Result<(), Error> {
        self._mlp.import_parameters(filename)
    }
}
