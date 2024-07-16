use std::str::FromStr;

use super::{
    char_set::CharSetError, CharSet, Forward, FunctionLayer, GradVal, GradValVec, LinearLayer, MLP
};
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
        let mut mlp = MLP::from_empty();

        for _ in 0..number_of_layers {
            mlp.add_layer(Box::new(LinearLayer::from_rand(n_chars, n_chars, true)));
            mlp.add_layer(Box::new(FunctionLayer::new(&GradVal::sigmoid, "Sigmoid")));
        }

        Bigram {
            _data: data,
            _charset: chars,
            _mlp: mlp,
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
        for n in 0..cycles {
            let loss = self._mlp.decend_grad(
                &self.extract_correlations(self._data.get_training_block(data_block_size)),
                learning_rate,
            );
            println!("Cycle {n} loss: {:e}", loss.value());
        }
    }

    pub fn predict(
        &self,
        seed_string: &str,
        number_of_characters: usize,
    ) -> Result<String, CharSetError> {
        let mut s = seed_string.to_owned();
        if s.len() == 0 {
            panic!("Aborting, cannot extrapolate from empty string.")
        }
        let mut last_char = self._charset.encode(s.chars().last().unwrap())?;
        for _ in 0..number_of_characters {
            last_char = self._mlp.forward(&last_char);
            last_char.collapse();
            s.push(self._charset.decode(&last_char)?);
        }
        Ok(s)
    }
}
