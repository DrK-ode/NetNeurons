/// If we ever want to change to f32, it is easy
pub type FloatType = f64;
/// All nodes are treated as matrices of shape (rows, columns)
pub type NodeShape = (usize, usize);

/// Mostly used for debug output
#[derive(Debug, PartialEq)]
pub enum VecOrientation {
    Column,
    Row,
}

/// Tensors of higher dimensions than matrices are not supported
#[derive(Debug, PartialEq)]
pub enum NodeType {
    None,
    Scalar,
    Vector(VecOrientation),
    Matrix,
}
