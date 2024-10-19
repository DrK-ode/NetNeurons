mod layers;
pub mod loss_functions;
mod multilayer;
mod traits;

pub use layers::{FunctionLayer, LinearLayer, ReshapeLayer};
pub use multilayer::MultiLayer;
pub use traits::{Layer, Parameters};
