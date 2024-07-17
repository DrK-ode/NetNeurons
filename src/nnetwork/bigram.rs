use std::{str::FromStr, time::Instant};

use super::{
    char_set::CharSetError, mlp::VectorFunctionLayer, CharSet, ElementFunctionLayer, Forward,
    GradVal, GradValVec, LinearLayer, MLP,
};
use crate::{data_set::DataSet, nnetwork::Parameters};

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

        for i in 1..=number_of_layers {
            mlp.add_layer(Box::new(LinearLayer::from_rand(n_chars, n_chars, true)));
            if i != number_of_layers {
                mlp.add_layer(Box::new(ElementFunctionLayer::new(
                    &GradVal::sigmoid,
                    "Sigmoid",
                )));
            }
        }
        mlp.add_layer(Box::new(VectorFunctionLayer::new(
            &GradValVec::soft_max,
            "SoftMax",
        )));

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

    pub fn learn(
        &mut self,
        cycles: usize,
        learning_rate: f32,
        data_block_size: usize,
        regularization: Option<f32>,
    ) {
        let timer = Instant::now();
        for n in 0..cycles {
            let training_data = self._data.get_training_block(data_block_size);
            let input_pairs = self.extract_correlations(training_data);

            let fit_loss: GradVal = input_pairs
                .iter()
                .map(|(inp, truth)| self._mlp.forward(&inp).maximum_likelihood(truth))
                .sum::<GradVal>()
                / (data_block_size as f32).into();

            let reg_loss = if regularization.is_some() {
                let regularization = regularization.unwrap();
                if regularization < 0. {
                    panic!("Regularization coefficient must be positive.");
                }
                let n_param = GradVal::from(self._mlp.parameters().count() as f32);
                // Mean of the sum of the squares of all parameters
                self._mlp
                    .parameters()
                    .map(|p| p.powf(2.))
                    .collect::<GradValVec>()
                    .sum()
                    * regularization.into()
                    / n_param
            } else {
                0f32.into()
            };

            let mut loss = fit_loss + reg_loss;

            self._mlp.decend_grad(&mut loss, learning_rate);
            println!("Cycle {n} loss: {:e}", loss.value());
        }
        println!("Trained network with {} parameters for {cycles} cycles in {} ms.", self._mlp.parameters().count(), timer.elapsed().as_millis() );
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
            last_char = self._mlp.forward(&last_char).collapsed();
            s.push(self._charset.decode(&last_char)?);
        }
        Ok(s)
    }
}
