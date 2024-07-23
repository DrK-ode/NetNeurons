use std::fmt::Display;

use crate::nnetwork::calculation_nodes::{FloatType, NetworkCalculation, TensorShared};

use super::neural_traits::{Layer, Parameters};

pub struct MLP {
    _layers: Vec<Box<dyn Layer>>,
    _calc: Option<NetworkCalculation>,
}

impl MLP {
    pub fn from_empty() -> MLP {
        MLP {
            _layers: Vec::new(),
            _calc: None,
        }
    }

    pub fn add_layer(&mut self, layer: Box<dyn Layer>) {
        self._layers.push(layer);
    }

    pub fn decend_grad(&mut self, learning_rate: FloatType) {
        self.parameters().for_each(|p| p.decend_grad(learning_rate));
    }

    pub fn forward_through_layers(&mut self, inp: &TensorShared) -> TensorShared {
        let mut out = inp.clone();

        for l in &self._layers {
            out = l.forward(&out);
        }
        out
    }

    /*pub fn perform_calculation(&self) {
    let timer = Instant::now();
    self._calc = Some(NetworkCalculation::new(&out));
    println!("Topological sorting took {} µs", timer.elapsed().as_micros());
    assert!(self._calc.is_some(), "Cannot perform calculation before it is constructed");
    let calc = self._calc.as_ref().unwrap();
    let timer = Instant::now();
    self._calc_result = Some(calc.evaluate());
    println!("Performing calculation took {} µs", timer.elapsed().as_micros());
    let timer = Instant::now();
    calc.back_propagation();
    println!("Back propagation took {} µs", timer.elapsed().as_micros());
    }*/
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

impl Parameters for MLP {
    fn parameters(&self) -> Box<dyn Iterator<Item = &TensorShared> + '_> {
        Box::new(self._layers.iter().map(|l| l.parameters()).flatten())
    }
}

#[cfg(test)]
mod tests {
    use crate::nnetwork::{LinearLayer, Neuron};

    use super::*;

    #[test]
    fn mlp_forward() {
        let mut mlp = MLP::from_empty();
        mlp.add_layer(Box::new(LinearLayer::from_tensors(
            TensorShared::from_vector(vec![1., 1., 1., 1., 1., 1.], (3, 2, 1)),
            None,
        )));
        mlp.add_layer(Box::new(LinearLayer::from_tensors(
            TensorShared::from_vector(vec![1., 1., 1., 1., 1., 1., 1., 1., 1.], (3, 3, 1)),
            None,
        )));
        mlp.add_layer(Box::new(LinearLayer::from_tensors(
            TensorShared::from_vector(vec![1., 1., 1., 1., 1., 1.], (2, 3, 1)),
            None,
        )));
        let inp = TensorShared::from_vector(vec![1., 2.], (2, 1, 1));
        let output = mlp.forward_through_layers(&inp);
        println!("{inp}, {output}");
        assert_eq!(output.value_as_col_vector().unwrap(), vec![27., 27.]);
    }
}
