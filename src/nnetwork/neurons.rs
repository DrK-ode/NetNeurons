use std::iter::empty;

use super::GradVal;
use rand::prelude::*;
use rand_distr::StandardNormal;

pub trait Forward {
    fn forward(&self, x: &Vec<GradVal>) -> Vec<GradVal>;
}

pub trait Parameters {
    fn parameters(&self) -> Box<dyn Iterator<Item = &GradVal> + '_> {
        Box::new(empty::<&GradVal>())
    }
}

pub trait Layer: Forward + Parameters {
    fn neurons(&self) -> Option<&Vec<Neuron>> {
        None
    }
    fn size_in(&self) -> Option<usize> {
        self.neurons().and_then(|n| Some(n[0].size()))
    }
    fn size_out(&self) -> Option<usize> {
        self.neurons().and_then(|n| Some(n.len()))
    }
}

pub struct Neuron {
    _w: Vec<GradVal>,
    _b: Option<GradVal>,
}

impl Neuron {
    pub fn from_rand(n_in: usize, biased: bool) -> Neuron {
        let mut neuron = Neuron {
            _w: Vec::with_capacity(n_in),
            _b: if biased {
                Some(f32::into(thread_rng().sample(StandardNormal)))
            } else {
                None
            },
        };
        for _ in 0..n_in {
            neuron
                ._w
                .push(GradVal::from(rand::thread_rng().gen::<f32>()));
        }
        neuron
    }

    pub fn from_vec(w: &Vec<f32>, b: Option<f32>) -> Neuron {
        Neuron {
            _w: w.iter().map(|w| GradVal::from(*w)).collect(),
            _b: if let Some(b) = b {
                Some(b.into())
            } else {
                None
            },
        }
    }

    pub fn from_value(w: f32, size: usize, b: Option<f32>) -> Neuron {
        Neuron {
            _w: (0..size).map(|_| GradVal::from(w)).collect(),
            _b: if let Some(b) = b {
                Some(b.into())
            } else {
                None
            },
        }
    }

    pub fn size(&self) -> usize {
        self._w.len()
    }

    pub fn eval(&self, prev: &Vec<GradVal>) -> GradVal {
        let mut result = self._w.iter().zip(prev.iter()).map(|(w, p)| w * p).sum();
        if self._b.is_some() {
            result = &result + self._b.as_ref().unwrap();
        }
        result
    }

    pub fn parameters(&self) -> Box<dyn Iterator<Item = &GradVal> + '_> {
        if self._b.is_some() {
            Box::new(
                self._w
                    .iter()
                    .chain(std::iter::once(self._b.as_ref().unwrap())),
            )
        } else {
            Box::new(self._w.iter())
        }
    }
}

pub struct LinearLayer {
    _neurons: Vec<Neuron>,
}

impl LinearLayer {
    pub fn from_rand(n_in: usize, n_out: usize, biased: bool) -> LinearLayer {
        LinearLayer {
            _neurons: (0..n_out)
                .map(|_| Neuron::from_rand(n_in, biased))
                .collect(),
        }
    }
    pub fn from_vec(neurons: Vec<Neuron>) -> LinearLayer {
        assert!(neurons.len() > 0, "Cannot create empty layer.");
        let l = neurons[0].size();
        assert!(
            neurons.iter().all(|n| n.size() == l),
            "All neurons in a layer must have equal size."
        );
        LinearLayer { _neurons: neurons }
    }
}

impl Forward for LinearLayer {
    fn forward(&self, prev: &Vec<GradVal>) -> Vec<GradVal> {
        self._neurons.iter().map(|n| n.eval(prev)).collect()
    }
}
impl Parameters for LinearLayer {
    fn parameters(&self) -> Box<dyn Iterator<Item = &GradVal> + '_> {
        Box::new(self._neurons.iter().map(|n| n.parameters()).flatten())
    }
}
impl Layer for LinearLayer {
    fn neurons(&self) -> Option<&Vec<Neuron>> {
        Some(&self._neurons)
    }
}

pub struct FunctionLayer {
    _func: &'static dyn Fn(&GradVal) -> GradVal,
}

impl FunctionLayer {
    pub fn new(f: &'static dyn Fn(&GradVal) -> GradVal) -> FunctionLayer {
        FunctionLayer { _func: f }
    }
}

impl Forward for FunctionLayer {
    fn forward(&self, x: &Vec<GradVal>) -> Vec<GradVal> {
        x.iter().map(|n| (self._func)(n)).collect()
    }
}
impl Parameters for FunctionLayer {
    fn parameters(&self) -> Box<dyn Iterator<Item = &GradVal> + '_> {
        Box::new(empty::<&GradVal>())
    }
}
impl Layer for FunctionLayer {}

pub struct MLP {
    _layers: Vec<Box<dyn Layer>>,
}

impl MLP {
    pub fn from_empty() -> MLP {
        MLP {
            _layers: Vec::new(),
        }
    }

    pub fn new(n_layer: usize, layer_size: usize, n_in: usize, n_out: usize) -> MLP {
        let mut mlp = MLP {
            _layers: Vec::with_capacity(n_layer),
        };
        for i in 0..n_layer {
            let n_in = if i == 0 { n_in } else { layer_size };
            let n_out = if i == n_layer - 1 { n_out } else { layer_size };
            mlp._layers
                .push(Box::new(LinearLayer::from_rand(n_in, n_out, true)));
            mlp._layers
                .push(Box::new(FunctionLayer::new(&GradVal::sigmoid)));
        }
        mlp
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
}

impl Forward for MLP {
    fn forward(&self, prev: &Vec<GradVal>) -> Vec<GradVal> {
        self._layers
            .iter()
            .fold(prev.to_vec(), |a, b| b.forward(&a))
    }
}
impl Parameters for MLP {
    fn parameters(&self) -> Box<dyn Iterator<Item = &GradVal> + '_> {
        Box::new(self._layers.iter().map(|l| l.parameters()).flatten())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unbiased_layer_forward() {
        let mut layer = LinearLayer::from_rand(2, 2, false);
        layer._neurons[0]._w = vec![GradVal::from(1.0), GradVal::from(2.0)];
        layer._neurons[1]._w = vec![GradVal::from(4.0), GradVal::from(5.0)];
        let input = vec![GradVal::from(1.0), GradVal::from(2.0)];
        let output = layer.forward(&input);
        assert_eq!(
            output,
            vec![GradVal::from(1. + 4.), GradVal::from(4. + 10.)]
        );
    }

    #[test]
    fn biased_layer_forward() {
        let mut layer = LinearLayer::from_rand(2, 2, true);
        layer._neurons[0]._w = vec![GradVal::from(1.0), GradVal::from(2.0)];
        layer._neurons[1]._w = vec![GradVal::from(4.0), GradVal::from(5.0)];
        layer._neurons[0]._b = Some(GradVal::from(3.0));
        layer._neurons[1]._b = Some(GradVal::from(6.0));
        let input = vec![GradVal::from(1.0), GradVal::from(2.0)];
        let output = layer.forward(&input);
        assert_eq!(
            output,
            vec![GradVal::from(1. + 4. + 3.), GradVal::from(4. + 10. + 6.)]
        );
    }

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
        let input = vec![GradVal::from(1.0), GradVal::from(2.0)];
        let output = mlp.forward(&input);
        assert_eq!(output, vec![GradVal::from(27.0), GradVal::from(27.0)]);
    }
}
