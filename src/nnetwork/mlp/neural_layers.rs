use std::fmt::Display;

use crate::nnetwork::{GradVal, GradValVec};

use super::{
    neural_traits::{Forward, Parameters},neurons::Neuron,
};

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
        GradValVec::from(
            self._neurons
                .iter()
                .map(|n| n.forward(prev))
                .collect::<Vec<_>>(),
        )
    }
}

impl Parameters for LinearLayer {
    fn parameters(&mut self) -> Box<dyn Iterator<Item = &mut GradVal> + '_> {
        Box::new(self._neurons.iter_mut().map(|n| n.parameters()).flatten())
    }
}

pub trait Layer: Forward<Output = GradValVec> + Parameters + Display {
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

impl Layer for LinearLayer {
    fn neurons(&self) -> Option<&Vec<Neuron>> {
        Some(&self._neurons)
    }
}

pub struct ElementFunctionLayer {
    _func: &'static dyn Fn(&GradVal) -> GradVal,
    _label: String,
}

impl ElementFunctionLayer {
    pub fn new(f: &'static dyn Fn(&GradVal) -> GradVal, label: &str) -> ElementFunctionLayer {
        ElementFunctionLayer {
            _func: f,
            _label: label.into(),
        }
    }
}

impl Display for ElementFunctionLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "ElementFunctionLayer: [{:?}]", self._label)
    }
}

impl Forward for ElementFunctionLayer {
    type Output = GradValVec;
    fn forward(&self, x: &GradValVec) -> Self::Output {
        GradValVec::from(x.iter().map(|n| (self._func)(n)).collect::<Vec<_>>())
    }
}
impl Parameters for ElementFunctionLayer {}
impl Layer for ElementFunctionLayer {}

pub struct VectorFunctionLayer {
    _func: &'static dyn Fn(&GradValVec) -> GradValVec,
    _label: String
}

impl VectorFunctionLayer {
    pub fn new(f: &'static dyn Fn(&GradValVec) -> GradValVec, label: &str) -> VectorFunctionLayer {
        VectorFunctionLayer{
            _func: f,
            _label: label.into()
        }
    }
}

impl Display for VectorFunctionLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "VectorFunctionLayer: [{:?}]", self._label)
    }
}

impl Forward for VectorFunctionLayer {
    type Output = GradValVec;
    fn forward(&self, x: &GradValVec) -> Self::Output {
        (self._func)(x)
    }
}
impl Parameters for VectorFunctionLayer {}
impl Layer for VectorFunctionLayer {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unbiased_layer_forward() {
        let mut layer = LinearLayer::from_rand(2, 2, false);
        layer._neurons[0].set_weights(vec![GradVal::from(1.0), GradVal::from(2.0)]);
        layer._neurons[1].set_weights(vec![GradVal::from(4.0), GradVal::from(5.0)]);
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
        layer._neurons[0].set_weights(vec![GradVal::from(1.0), GradVal::from(2.0)]);
        layer._neurons[1].set_weights(vec![GradVal::from(4.0), GradVal::from(5.0)]);
        layer._neurons[0].set_bias( Some(GradVal::from(3.0)));
        layer._neurons[1].set_bias(Some(GradVal::from(6.0)));
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
}
