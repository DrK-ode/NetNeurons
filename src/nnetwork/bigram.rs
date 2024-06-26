use std::str::FromStr;

use super::{CharSet, GradValVec, MLP};
use crate::data_set::DataSet;

pub struct Bigram {
    _data: DataSet,
    _charset: CharSet,
    _mlp: MLP,
}

impl Bigram {
    pub fn new(data: DataSet, number_of_layers: usize) -> Self {
        let chars = CharSet::from_str(data.get_training_data()).unwrap();
        let n_chars = chars.size();
        Bigram {
            _data: data,
            _charset: chars,
            _mlp: MLP::new(number_of_layers, n_chars, n_chars, n_chars),
        }
    }

    fn extract_correlations(&self, data_block: &str) -> Vec<(GradValVec, GradValVec)> {
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

    pub fn learn(&mut self, cycles: usize, learning_rate: f32, data_block_size: usize) {
        self._mlp.decend_grad(
            &self.extract_correlations(self._data.get_training_block(data_block_size)),
            cycles,
            learning_rate,
        );
    }
}
