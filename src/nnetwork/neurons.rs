use std::{
    fmt::Display,
    iter::empty, ops::Div,
};

use super::{gradval::GradValVec, GradVal};
use rand::prelude::*;
use rand_distr::StandardNormal;

pub trait Forward {
    type Output;
    fn forward(&self, x: &GradValVec) -> Self::Output;
}

pub trait Parameters {
    fn parameters(&mut self) -> Box<dyn Iterator<Item = &mut GradVal> + '_> {
        Box::new(empty::<&mut GradVal>())
    }
}

pub trait Layer: Forward<Output=GradValVec> + Parameters + Display {
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
                .push(f32::into(thread_rng().sample(StandardNormal)));
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

    pub fn parameters(&mut self) -> Box<dyn Iterator<Item = &mut GradVal> + '_> {
        if self._b.is_some() {
            Box::new(
                self._w
                    .iter_mut()
                    .chain(std::iter::once(self._b.as_mut().unwrap())),
            )
        } else {
            Box::new(self._w.iter_mut())
        }
    }
}

impl Forward for Neuron{
    type Output = GradVal;
    fn forward(&self, prev: &GradValVec) -> Self::Output {
        let mut result = self._w.iter().zip(prev.iter()).map(|(w, p)| w * p).sum();
        if self._b.is_some() {
            result = &result + self._b.as_ref().unwrap();
        }
        result
    }
}

impl Display for Neuron {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for w in &self._w {
            write!(f, "{w}, ")?;
        }
        if let Some(ref bias) = self._b {
            write!(f, "bias: {bias}")?;
        }
        writeln!(f, "]")
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

impl Display for LinearLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "LinearLayer: [")?;
        for n in &self._neurons {
            n.fmt(f)?;
        }
        writeln!(f, "]")
    }
}

impl Forward for LinearLayer {
    type Output = GradValVec;
    fn forward(&self, prev: &GradValVec) -> Self::Output {
        GradValVec::from(self._neurons.iter().map(|n| n.forward(prev)).collect::<Vec<_>>())
    }
}
impl Parameters for LinearLayer {
    fn parameters(&mut self) -> Box<dyn Iterator<Item = &mut GradVal> + '_> {
        Box::new(self._neurons.iter_mut().map(|n| n.parameters()).flatten())
    }
}
impl Layer for LinearLayer {
    fn neurons(&self) -> Option<&Vec<Neuron>> {
        Some(&self._neurons)
    }
}

pub struct FunctionLayer {
    _func: &'static dyn Fn(&GradVal) -> GradVal,
    _label: String,
}

impl FunctionLayer {
    pub fn new(f: &'static dyn Fn(&GradVal) -> GradVal, label: &str) -> FunctionLayer {
        FunctionLayer {
            _func: f,
            _label: label.into(),
        }
    }
}

impl Display for FunctionLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "FunctionLayer: [{:?}]", self._label)
    }
}

impl Forward for FunctionLayer {
    type Output = GradValVec;
    fn forward(&self, x: &GradValVec) -> Self::Output {
        GradValVec::from(x.iter().map(|n| (self._func)(n)).collect::<Vec<_>>())
    }
}
impl Parameters for FunctionLayer {}
impl Layer for FunctionLayer {}

pub enum LossType {
    MaximumLikelihood,
    LeastSquare,
}


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
                .push(Box::new(FunctionLayer::new(&GradVal::sigmoid, "Sigmoid")));
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

    fn maximum_likelihood(output: &GradValVec, truth: &GradValVec) -> GradVal {
        let exped: GradValVec = output.iter().map(|v| v.exp()).collect();
        let norm = exped.sum() / (exped.size() as f32).into();
        exped
            .iter()
            .map(|v| v.div(&norm))
            .zip(truth.iter())
            .map(|(ref p, t)| (p * t).log())
            .sum::<GradVal>()
            .div((truth.size() as f32).into())
    }

    fn least_squares(output: &GradValVec, truth: &GradValVec) -> GradVal {
        output
                .iter()
                .zip(truth.iter())
                .map(|(v, t)| (v - t).powf(2.))
                .sum::<GradVal>()
                .div((truth.size() as f32).into())
    }

    pub fn loss(output: &GradValVec, truth: &GradValVec, formula: LossType) -> GradVal {
        assert_eq!(
            output.size(),
            truth.size(),
            "Cannot compare non-equal sized NnVec"
        );
        match formula {
            LossType::MaximumLikelihood => MLP::maximum_likelihood(output, truth),
            LossType::LeastSquare => MLP::least_squares(output, truth),
        }
    }

    pub fn decend_grad(&mut self, input_pairs: &Vec<(GradValVec,GradValVec)>, cycles: usize, learning_rate: f32) {
        for _ in 0..cycles {
            let mut losses: GradVal = input_pairs
                .iter()
                .map(|(inp, truth)| {
                    let out = self.forward(&inp);
                    MLP::loss(&out, &truth, LossType::MaximumLikelihood)
                })
                .sum();
            losses.backward();

            self.parameters().for_each(|p: &mut GradVal| {
                p.set_value(p.value() - learning_rate * p.grad().unwrap());
            })
        }
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
    use super::*;

    #[test]
    fn unbiased_layer_forward() {
        let mut layer = LinearLayer::from_rand(2, 2, false);
        layer._neurons[0]._w = vec![GradVal::from(1.0), GradVal::from(2.0)];
        layer._neurons[1]._w = vec![GradVal::from(4.0), GradVal::from(5.0)];
        let input = GradValVec::from(vec![GradVal::from(1.0), GradVal::from(2.0)]);
        let output = layer.forward(&input);
        assert_eq!(
            output,
            GradValVec::from(vec![GradVal::from(1. + 4.), GradVal::from(4. + 10.)])
        );
    }

    #[test]
    fn biased_layer_forward() {
        let mut layer = LinearLayer::from_rand(2, 2, true);
        layer._neurons[0]._w = vec![GradVal::from(1.0), GradVal::from(2.0)];
        layer._neurons[1]._w = vec![GradVal::from(4.0), GradVal::from(5.0)];
        layer._neurons[0]._b = Some(GradVal::from(3.0));
        layer._neurons[1]._b = Some(GradVal::from(6.0));
        let input = GradValVec::from(vec![GradVal::from(1.0), GradVal::from(2.0)]);
        let output = layer.forward(&input);
        assert_eq!(
            output,
            GradValVec::from(vec![
                GradVal::from(1. + 4. + 3.),
                GradVal::from(4. + 10. + 6.)
            ])
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
        let input = GradValVec::from(vec![GradVal::from(1.0), GradVal::from(2.0)]);
        let output = mlp.forward(&input);
        println!("{input}, {output}");
        assert_eq!(
            output,
            GradValVec::from(vec![GradVal::from(27.0), GradVal::from(27.0)])
        );
    }
}
