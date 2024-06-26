pub mod bigram;
pub mod char_set;
pub mod gradval;
pub mod neurons;

pub use bigram::Bigram;
pub use char_set::CharSet;
pub use gradval::{GradVal,GradValVec};
pub use neurons::{Neuron,LinearLayer,FunctionLayer,MLP,Forward};