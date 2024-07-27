mod multi_layer;
mod layer;
mod layer_traits;

pub use multi_layer::MultiLayer;
pub use layer::{FunctionLayer, LinearLayer};
pub use layer_traits::{Forward, Parameters, Layer};