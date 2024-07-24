mod multi_layer;
mod neural_layers;
mod neural_traits;

pub use multi_layer::MultiLayer;
pub use neural_layers::{FunctionLayer, LinearLayer};
pub use neural_traits::{Forward, Parameters};