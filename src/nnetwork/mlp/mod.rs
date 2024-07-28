mod multilayer;
mod layer;
mod layer_traits;

pub use multilayer::Predictor;
pub use multilayer::Trainer;
pub use layer::{FunctionLayer, LinearLayer, ReshapeLayer};
pub use layer_traits::{Forward, Parameters, Layer};