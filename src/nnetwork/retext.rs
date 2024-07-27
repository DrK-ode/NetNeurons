use std::time::Instant;

use crate::{
    data_preparing::data_set::{DataSet, DataSetError},
    nnetwork::Parameters,
};

use super::{FloatType, FunctionLayer, Layer, LinearLayer, MultiLayer, TensorShared};

pub struct ReText {
    _dataset: DataSet,
    _ml: MultiLayer,
    _block_size: usize,
}

impl ReText {
    pub fn new(
        data: DataSet,
        batch_size: usize,
        data_block_size: usize,
        embed_dim: Option<usize>,
        n_hidden_layers: usize,
        layer_dim: usize,
        regularization: Option<FloatType>,
    ) -> ReText {
        let n_chars = data.number_of_chars();
        let mut layers: Vec<Box<dyn Layer>> = Vec::new();
        let non_linearity = FunctionLayer::new(&FunctionLayer::tanh, "Tanh");
        const BIASED_LAYERS: bool = true;

        // Hidden layers
        for _ in 0..n_hidden_layers {
            layers.push(Box::new(LinearLayer::from_rand(
                layer_dim,
                layer_dim,
                BIASED_LAYERS,
            )));
            layers.push(Box::new(non_linearity.clone()));
        }

        // Deembed
        layers.push(Box::new(LinearLayer::from_rand(
            layer_dim,
            n_chars,
            BIASED_LAYERS,
        )));

        layers.push(Box::new(FunctionLayer::new(
            &FunctionLayer::softmax,
            "SoftMax",
        )));

        let ml = MultiLayer::new_trainable(
            (n_chars, data_block_size, 1),
            embed_dim,
            (n_chars, 1, 1),
            batch_size,
            layers,
            regularization,
            &MultiLayer::neg_log_likelihood,
        );

        ReText {
            _dataset: data,
            _ml: ml,
            _block_size: data_block_size,
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
            let loss = self._ml.train(&correlations, learning_rate);
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
            self._ml.parameters().map(|p| p.len()).sum::<usize>(),
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
            last = MultiLayer::collapse(&self._ml.forward(&last));
            s.push(self._dataset.decode(&last)?);
        }
        Ok(s)
    }

    pub fn export_parameters(&self, filename: &str) -> std::io::Result<String> {
        self._ml.export_parameters(filename)
    }

    pub fn import_parameters(&mut self, filename: &str) -> std::io::Result<()> {
        self._ml.import_parameters(filename)
    }
}
