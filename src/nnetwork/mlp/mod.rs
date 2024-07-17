mod neurons;
mod neural_traits;
mod neural_layers;
mod mlp;

pub use neurons::Neuron;
pub use neural_traits::{Forward,Parameters};
pub use neural_layers::{ElementFunctionLayer,VectorFunctionLayer,LinearLayer};
pub use mlp::MLP;