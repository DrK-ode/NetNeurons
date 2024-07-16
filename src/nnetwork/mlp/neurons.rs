use std::fmt::Display;

use rand::prelude::*;
use rand_distr::StandardNormal;

use crate::nnetwork::{GradVal, GradValVec};

use super::neural_traits::Forward;

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

    pub fn set_weights(&mut self, w: Vec<GradVal>) {
        self._w = w;
    }

    pub fn set_bias(&mut self, b: Option<GradVal>){
        self._b = b;
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

impl Forward for Neuron {
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