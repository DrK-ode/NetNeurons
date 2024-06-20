pub mod char_set;
pub mod gradval;
pub mod neurons;

pub use char_set::CharSet;
pub use gradval::GradVal;
pub use neurons::{Neuron,LinearLayer,FunctionLayer,MLP,Forward,NnVec};