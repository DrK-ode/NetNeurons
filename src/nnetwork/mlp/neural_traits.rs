use std::iter::empty;

use crate::nnetwork::{GradVal, GradValVec};

pub trait Forward {
    type Output;
    fn forward(&self, x: &GradValVec) -> Self::Output;
}

pub trait Parameters {
    fn parameters(&mut self) -> Box<dyn Iterator<Item = &mut GradVal> + '_> {
        Box::new(empty::<&mut GradVal>())
    }
}