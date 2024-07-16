use std::{fmt::Display, ops::Deref};

use rand::Rng;

use super::gradval::GradVal;

// GradVal vector type
#[derive(Debug, PartialEq)]
pub struct GradValVec {
    _values: Vec<GradVal>,
}

impl Deref for GradValVec {
    type Target = Vec<GradVal>;

    fn deref(&self) -> &Self::Target {
        &self._values
    }
}

impl Display for GradValVec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "NnVec: [")?;
        for v in &self._values {
            writeln!(f, "{v}")?;
        }
        writeln!(f, "]")
    }
}

impl From<Vec<GradVal>> for GradValVec {
    fn from(values: Vec<GradVal>) -> Self {
        GradValVec { _values: values }
    }
}

impl From<Vec<f32>> for GradValVec {
    fn from(values: Vec<f32>) -> Self {
        GradValVec { _values: values.iter().map(|&v| GradVal::from(v)).collect() }
    }
}

impl FromIterator<GradVal> for GradValVec {
    fn from_iter<T: IntoIterator<Item = GradVal>>(iter: T) -> Self {
        GradValVec {
            _values: iter.into_iter().collect(),
        }
    }
}

impl GradValVec {
    pub fn size(&self) -> usize {
        self._values.len()
    }

    pub fn sum(&self) -> GradVal {
        GradVal::sum(&self._values)
    }

    pub fn normalize(&mut self){
        let norm = self.sum();
        self._values = self._values.iter().map(|v| v / &norm ).collect();
    }

    pub fn collapse(&mut self) -> &Self {
        self.normalize();
        let mut vec = vec![GradVal::from(0.); self.len()];
        let mut rnd = rand::thread_rng().gen_range(0f32..1f32);
        for (i,v) in self._values.iter().enumerate() {
            rnd -= v.value();
            if rnd <= 0. {
                vec[i] = GradVal::from(1.);
                break; 
            }
        }
        self._values = vec;
        self
    }
}