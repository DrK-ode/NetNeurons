use std::{fmt::Display, iter::empty};

use crate::nnetwork::calculation_nodes::TensorShared;

pub trait Forward {
    fn forward(&self, inp: &TensorShared) -> TensorShared;
}

pub trait Parameters {
    fn parameters(&self) -> Box<dyn Iterator<Item = &TensorShared> + '_> {
        Box::new(empty::<&TensorShared>())
    }
}

pub trait Layer: Forward + Parameters + Display {}
