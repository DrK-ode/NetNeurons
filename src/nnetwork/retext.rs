use std::time::Instant;

use crate::{
    data_preparing::data_set::{DataSet, DataSetError},
    nnetwork::{Parameters, ReshapeLayer},
};

use super::{
    FloatType, Forward, FunctionLayer, Layer, LinearLayer, Predictor, TensorShared, Trainer,
};

pub struct ReText {
    _dataset: DataSet,
    _trainer: Trainer,
    _predictor: Predictor,
    _block_size: usize,
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
            let reshape_layer =
                ReshapeLayer::new((block_size * embed_dim, 1, 1), "Reshaping layer");
            let resize_layer = LinearLayer::from_rand(
                layer_dim,
                block_size * embed_dim,
                BIASED_LAYERS,
                "Resizing layer (in)",
            );
            layers.push(Box::new(embed_layer));
            layers.push(Box::new(reshape_layer));
            layers.push(Box::new(resize_layer));
        } else {
            let resize_layer =
                LinearLayer::from_rand(layer_dim, n_chars, BIASED_LAYERS, "Resizing layer (in)");
            layers.push(Box::new(resize_layer));
        }
        let regularize_layer = non_linearity.clone();
        layers.push(Box::new(regularize_layer));

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
        batch_size: usize,
        block_size: usize,
        embed_dim: Option<usize>,
        n_hidden_layers: usize,
        layer_dim: usize,
        regularization: Option<FloatType>,
    ) -> ReText {
        let n_chars = data.number_of_chars();
        let training_layers =
            Self::create_layers(n_chars, block_size, embed_dim, n_hidden_layers, layer_dim);
        let predicting_layers =
            Self::create_layers(n_chars, block_size, embed_dim, n_hidden_layers, layer_dim);
        ReText {
            _dataset: data,
            _trainer: Trainer::new(
                (n_chars, block_size, 1),
                (n_chars, 1, 1),
                batch_size,
                training_layers,
                regularization,
                &Trainer::neg_log_likelihood,
            ),
            _predictor: Predictor::new((n_chars, 1, 1), predicting_layers),
            _block_size: block_size,
        }
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
            let training_data = self._dataset.training_block(data_size);
            let correlations = self.extract_correlations(training_data);
            let timer = Instant::now();
            let loss = self._trainer.train(&correlations, learning_rate);
            if verbose {
                let width = (cycles as f64).log10() as usize;
                println!(
                    "Cycle #{n: >width$}: [ loss: {:.3e}, duration: {} µs ]",
                    loss.value_as_scalar().unwrap(),
                    timer.elapsed().as_micros()
                );
            }
        }
        println!(
            "Trained network with {} parameters for {cycles} cycles in {} ms.",
            self._trainer.parameters().map(|p| p.len()).sum::<usize>(),
            timer.elapsed().as_millis()
        );
    }

    // Returns a list of all correlations in the data encoded as a tuple of Matrix(m*n) and ColumnVector(n).
    fn extract_correlations(&self, data: &str) -> Vec<(TensorShared, TensorShared)> {
        data.char_indices()
            .zip(data.char_indices().skip(self._block_size))
            .map(|((i, _prev), (j, next))| {
                let prev = &data[i..j];
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

    pub fn predict(
        &mut self,
        seed_string: &str,
        number_of_characters: usize,
    ) -> Result<String, DataSetError> {
        let mut s = seed_string.to_owned();
        if s.len() < self._block_size {
            panic!(
                "Aborting, cannot extrapolate from string shorter than {}.",
                self._block_size
            );
        }
        let range = (s.len() - (self._block_size + 1))..;
        let mut last = self._dataset.encode(&s[range])?;
        for _ in 0..number_of_characters {
            last = Predictor::collapse(&self._predictor.forward(&last));
            s.push(self._dataset.decode(&last)?);
        }
        Ok(s)
    }

    pub fn export_parameters(&self, filename: &str) -> std::io::Result<String> {
        self._trainer.export_parameters(filename)
    }

    pub fn import_parameters(&mut self, filename: &str) -> std::io::Result<()> {
        self._trainer.import_parameters(filename)
    }
}
