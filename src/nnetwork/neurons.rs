use super::GradVal;
use rand::Rng;

pub struct Neuron {
    _w: Vec<GradVal>,
    _b: GradVal,
}

impl Neuron {
    pub fn new(n_in: usize) -> Neuron {
        let mut neuron = Neuron {
            _w: Vec::with_capacity(n_in),
            _b: GradVal::from(rand::thread_rng().gen::<f32>()),
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

    pub fn parameters(&self) -> (&Vec<GradVal>, &GradVal) {
        (&self._w, &self._b)
    }
}

pub struct NeuronLayer {
    _neurons: Vec<Neuron>,
}

impl NeuronLayer {
    pub fn new(n_in: usize, n_out: usize) -> NeuronLayer {
        let mut layer = NeuronLayer {
            _neurons: Vec::with_capacity(n_out),
        };
        for _ in 0..n_out {
            layer._neurons.push(Neuron::new(n_in));
        }
        layer
    }

    pub fn eval(&self, prev: &Vec<GradVal>) -> Vec<GradVal> {
        self._neurons.iter().map(|n| n.eval(prev)).collect()
    }

    pub fn parameters(&self) -> Vec<(&Vec<GradVal>, &GradVal)> {
        self._neurons.iter().map(|n| n.parameters()).collect()
    }
}
