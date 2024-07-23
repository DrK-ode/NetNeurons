mod mlp;
mod neural_layers;
mod neural_traits;

pub use mlp::MLP;
pub use neural_layers::{FunctionLayer, LinearLayer};
pub use neural_traits::{Forward, Parameters};