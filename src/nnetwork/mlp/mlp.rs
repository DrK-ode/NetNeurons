use std::{fmt::Display, time::Instant};

use crate::nnetwork::{GradVal, GradValVec};

use super::{neural_layers::Layer, neural_traits::Forward, neural_traits::Parameters};

pub struct MLP {
    _layers: Vec<Box<dyn Layer>>,
}

impl MLP {
    pub fn from_empty() -> MLP {
        MLP {
            _layers: Vec::new(),
        }
    }

    pub fn from_vec(layers: Vec<Box<dyn Layer>>) -> MLP {
        if layers.len() > 0 {
            return MLP::from_empty();
        }
        Self::check_layers(&layers);
        MLP { _layers: layers }
    }

    fn check_layers(layers: &Vec<Box<dyn Layer>>) {
        let mut nin: Option<usize> = None;
        for (i, l) in layers.iter().enumerate() {
            if let Some(n2) = l.size_in() {
                if let Some(n1) = nin {
                    assert_eq!(
                        n1, n2,
                        "Layer with index {i} should have had size {n1} but {n2} was found."
                    );
                }
                nin = l.size_out();
            }
        }
    }

    pub fn add_layer(&mut self, layer: Box<dyn Layer>) {
        self._layers.push(layer);
        Self::check_layers(&self._layers);
    }

    pub fn decend_grad(&mut self, loss: &mut GradVal, learning_rate: f32) {
        let backward_timer = Instant::now();
        loss.backward();
        println!("Backward propagation took {} ms.", backward_timer.elapsed().as_millis());
        self.parameters().for_each(|p: &mut GradVal| {
            p.set_value(p.value() - learning_rate * p.grad().unwrap());
        });
    }
}

impl Display for MLP {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "MLP: [")?;
        for layer in &self._layers {
            layer.fmt(f)?;
        }
        writeln!(f, "]")
    }
}

impl Forward for MLP {
    type Output = GradValVec;
    fn forward(&self, prev: &GradValVec) -> GradValVec {
        let mut output = Vec::<f32>::into(Vec::new());
        let mut input = prev;

        for l in &self._layers {
            output = l.forward(input);
            input = &output;
        }
        return output;
    }
}
impl Parameters for MLP {
    fn parameters(&mut self) -> Box<dyn Iterator<Item = &mut GradVal> + '_> {
        Box::new(self._layers.iter_mut().map(|l| l.parameters()).flatten())
    }
}

#[cfg(test)]
mod tests {
    use crate::nnetwork::{LinearLayer, Neuron};

    use super::*;

    #[test]
    fn mlp_forward() {
        let mut mlp = MLP::from_empty();
        mlp.add_layer(Box::new(LinearLayer::from_vec(vec![
            Neuron::from_value(1., 2, None),
            Neuron::from_value(1., 2, None),
            Neuron::from_value(1., 2, None),
        ])));
        mlp.add_layer(Box::new(LinearLayer::from_vec(vec![
            Neuron::from_value(1., 3, None),
            Neuron::from_value(1., 3, None),
            Neuron::from_value(1., 3, None),
        ])));
        mlp.add_layer(Box::new(LinearLayer::from_vec(vec![
            Neuron::from_value(1., 3, None),
            Neuron::from_value(1., 3, None),
        ])));
        let input = GradValVec::from(vec![GradVal::from(1.0), GradVal::from(2.0)]);
        let output = mlp.forward(&input);
        println!("{input}, {output}");
        assert_eq!(
            output,
            GradValVec::from(vec![GradVal::from(27.0), GradVal::from(27.0)])
        );
    }
}
