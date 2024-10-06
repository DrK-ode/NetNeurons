mod layer;
mod multilayer;
mod parameters;

pub use layer::{FunctionLayer, Layer, LinearLayer, ReshapeLayer};
pub use multilayer::MultiLayer;
pub use parameters::{Parameters,ParameterBundle};
