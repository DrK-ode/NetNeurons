mod multilayer;
mod layer;
mod parameters;

pub use multilayer::Predictor;
pub use multilayer::Trainer;
pub use layer::{FunctionLayer, LinearLayer, ReshapeLayer, Layer};
pub use parameters::{Parameters,ParameterBundle};