use std::iter::empty;

use super::GradVal;
use rand::prelude::*;
use rand_distr::StandardNormal;

pub trait Forward {
    fn forward(&self, x: &Vec<GradVal>) -> Vec<GradVal>;
    fn parameters(&self) -> Box<dyn Iterator<Item=&GradVal> + '_>;
}

pub struct BiasedNeuron {
    _w: Vec<GradVal>,
    _b: GradVal,
}

impl BiasedNeuron {
    pub fn new(n_in: usize) -> BiasedNeuron {
        let mut neuron = BiasedNeuron {
            _w: Vec::with_capacity(n_in),
            _b: f32::into(thread_rng().sample(StandardNormal)),
        };
        for _ in 0..n_in {
            neuron
                ._w
                .push(GradVal::from(rand::thread_rng().gen::<f32>()));
        }
        neuron
    }
    pub fn eval(&self, prev: &Vec<GradVal>) -> GradVal {
        &self._w.iter().zip(prev.iter()).map(|(w, p)| w * p).sum() + &self._b
    }

    pub fn parameters(&self) -> impl Iterator<Item=&GradVal> {
        self._w.iter().chain( std::iter::once(&self._b) )
    }
}

pub struct LinearLayer {
    _neurons: Vec<BiasedNeuron>,
}

impl LinearLayer {
    pub fn new(n_in: usize, n_out: usize) -> LinearLayer {
        let mut layer = LinearLayer {
            _neurons: Vec::with_capacity(n_out),
        };
        for _ in 0..n_out {
            layer._neurons.push(BiasedNeuron::new(n_in));
        }
        layer
    }
}

impl Forward for LinearLayer{
    fn forward(&self, prev: &Vec<GradVal>) -> Vec<GradVal> {
        self._neurons.iter().map(|n| n.eval(prev)).collect()
    }
    fn parameters(&self) -> Box<dyn Iterator<Item=&GradVal> + '_> {
        Box::new(self._neurons.iter().map(|n| n.parameters() ).flatten())
    }
}

pub struct FunctionLayer {
    _func: &'static dyn Fn(&GradVal) -> GradVal,
}

impl FunctionLayer{
    pub fn new( f: &'static dyn Fn(&GradVal) -> GradVal ) -> FunctionLayer {
        FunctionLayer{
        _func: f,
        }
    }
}

impl Forward for FunctionLayer {
    fn forward(&self, x: &Vec<GradVal>) -> Vec<GradVal> {
        x.iter().map(|n| (self._func)(n)).collect()
    }
    fn parameters(&self) -> Box<dyn Iterator<Item=&GradVal> + '_> {
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
        for i in 0..n_layer{
            let n_in = if i == 0 {n_in} else {layer_size};
            let n_out = if i == n_layer-1 {n_out} else {layer_size};
            mlp._layers.push(Box::new(LinearLayer::new(n_in, n_out)));
        }
        mlp
    }
}

impl Forward for MLP {
    fn forward(&self, prev: &Vec<GradVal> ) -> Vec<GradVal> {
        self._layers.iter().fold(prev.to_vec(), |a,b| b.forward(&a) )
    }
    fn parameters(&self) -> Box<dyn Iterator<Item=&GradVal> + '_> {
        Box::new(self._layers.iter().map(|l| l.parameters()).flatten())
    }
}
