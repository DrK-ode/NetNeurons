use std::iter::empty;

use super::GradVal;
use rand::prelude::*;
use rand_distr::StandardNormal;

pub trait Forward {
    fn forward(&self, x: &Vec<GradVal>) -> Vec<GradVal>;
    fn parameters(&self) -> Box<dyn Iterator<Item = &GradVal> + '_>;
}

pub struct Neuron {
    _w: Vec<GradVal>,
    _b: Option<GradVal>,
}

impl Neuron {
    pub fn new(n_in: usize, biased: bool) -> Neuron {
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
    pub fn new(n_in: usize, n_out: usize, biased: bool) -> LinearLayer {
        let mut layer = LinearLayer {
            _neurons: Vec::with_capacity(n_out),
        };
        for _ in 0..n_out {
            layer._neurons.push(Neuron::new(n_in, biased));
        }
        layer
    }
}

impl Forward for LinearLayer {
    fn forward(&self, prev: &Vec<GradVal>) -> Vec<GradVal> {
        self._neurons.iter().map(|n| n.eval(prev)).collect()
    }
    fn parameters(&self) -> Box<dyn Iterator<Item = &GradVal> + '_> {
        Box::new(self._neurons.iter().map(|n| n.parameters()).flatten())
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
    fn parameters(&self) -> Box<dyn Iterator<Item = &GradVal> + '_> {
        Box::new(empty::<&GradVal>())
    }
}

pub struct MLP {
    _layers: Vec<Box<dyn Forward>>,
}

impl MLP {
    pub fn new(n_layer: usize, layer_size: usize, n_in: usize, n_out: usize) -> MLP {
        let mut mlp = MLP {
            _layers: Vec::with_capacity(n_layer),
        };
        for i in 0..n_layer {
            let n_in = if i == 0 { n_in } else { layer_size };
            let n_out = if i == n_layer - 1 { n_out } else { layer_size };
            mlp._layers.push(Box::new(LinearLayer::new(n_in, n_out, true)));
            mlp._layers.push(Box::new(FunctionLayer::new( &GradVal::sigmoid)) );
        }
        mlp
    }
}

impl Forward for MLP {
    fn forward(&self, prev: &Vec<GradVal>) -> Vec<GradVal> {
        self._layers
            .iter()
            .fold(prev.to_vec(), |a, b| b.forward(&a))
    }
    fn parameters(&self) -> Box<dyn Iterator<Item = &GradVal> + '_> {
        Box::new(self._layers.iter().map(|l| l.parameters()).flatten())
    }
}

#[cfg(test)]
mod tests{
    use super::*;

    #[test]
    fn layer_forward(){
        let mut layer = LinearLayer::new(2, 2, true);
        layer._neurons[0]._w = vec![GradVal::from(1.0), GradVal::from(2.0)];
        layer._neurons[1]._w = vec![GradVal::from(4.0), GradVal::from(5.0)];
        layer._neurons[0]._b = Some(GradVal::from(3.0));
        layer._neurons[1]._b = Some(GradVal::from(6.0));
        let input = vec![GradVal::from(1.0), GradVal::from(2.0)];
        let output = layer.forward(&input);
        assert_eq!(output, vec![GradVal::from(1.+4.+3.), GradVal::from(4.+10.+6.)]);
    }
}
