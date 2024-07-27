mod calculation_nodes;
mod mlp;
mod retext;

pub use retext::ReText;
pub use calculation_nodes::{FloatType, TensorShape, TensorShared, TensorType, VecOrientation};
pub use mlp::{Forward, FunctionLayer, Layer, LinearLayer, MultiLayer, Parameters};
