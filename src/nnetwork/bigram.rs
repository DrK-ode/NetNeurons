use std::{str::FromStr, time::Instant};

use super::{
    calculation_nodes::TensorShared, char_set::CharSetError, CharSet, FunctionLayer, LinearLayer,
    MultiLayer,
};
use crate::{
    data_set::DataSet,
    nnetwork::{calculation_nodes::FloatType, Parameters},
};

pub struct Bigram {
    _data: DataSet,
    _charset: CharSet,
    _mlp: MultiLayer,
}

impl Bigram {
    pub fn new(data: DataSet, number_of_layers: usize) -> Self {
        let chars = CharSet::from_str(data.get_training_data()).unwrap();
        let n_chars = chars.size();
        let mut mlp = MultiLayer::from_empty();

        for i in 1..=number_of_layers {
            mlp.add_layer(Box::new(LinearLayer::from_rand(n_chars, n_chars, false)));
            if i != number_of_layers {
                mlp.add_layer(Box::new(FunctionLayer::new(
                    &FunctionLayer::sigmoid,
                    "Sigmoid",
                )));
            }
        }
        mlp.add_layer(Box::new(FunctionLayer::new(
            &FunctionLayer::softmax,
            "SoftMax",
        )));

        Bigram {
            _data: data,
            _charset: chars,
            _mlp: mlp,
        }
    }

    fn extract_correlations(&self, data_block: &str) -> Vec<(TensorShared, TensorShared)> {
        data_block
            .chars()
            .zip(data_block.chars().skip(1))
            .map(|(prev, next)| {
                (
                    self._charset
                        .encode(prev)
                        .expect("Cannot encode character: {prev}"),
                    self._charset
                        .encode(next)
                        .expect("Cannot encode character: {next}"),
                )
            })
            .collect()
    }

    pub fn learn(
        &mut self,
        cycles: usize,
        learning_rate: FloatType,
        data_block_size: usize,
        regularization: Option<FloatType>,
        verbose: bool,
    ) {
        let timer = Instant::now();
        let inp_shape = (self._charset.size(), 1, 1);
        let out_shape = inp_shape;
        // Very wasteful but let's keep it for now
        let training_data = self._data.get_training_block(data_block_size);
        let correlations = self.extract_correlations(training_data);
        self._mlp.define_training(
            correlations.len(),
            inp_shape,
            out_shape,
            regularization,
            &MultiLayer::neg_log_likelihood,
        );
        for n in 0..cycles {
            let training_data = self._data.get_training_block(data_block_size);
            let correlations = self.extract_correlations(training_data);
            let loss = self._mlp.train(&correlations, learning_rate, verbose);
            if verbose {
                println!("Cycle {n} loss: {:e}", loss.value_as_scalar().unwrap())
            }
        }
        println!(
            "Trained network with {} parameters for {cycles} cycles in {} ms.",
            self._mlp.parameters().count(),
            timer.elapsed().as_millis()
        );
    }

    pub fn predict(
        &mut self,
        seed_string: &str,
        number_of_characters: usize,
    ) -> Result<String, CharSetError> {
        let mut s = seed_string.to_owned();
        if s.is_empty() {
            panic!("Aborting, cannot extrapolate from empty string.")
        }
        let mut last_char = self._charset.encode(s.chars().last().unwrap())?;
        self._mlp.define_forward(last_char.shape());
        for _ in 0..number_of_characters {
            last_char = MultiLayer::collapse(&self._mlp.forward(&last_char));
            s.push(self._charset.decode(&last_char)?);
        }
        Ok(s)
    }
}
