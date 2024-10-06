pub type FloatType = f64;
pub type NodeShape = (usize, usize);

// Debug
#[derive(Debug, PartialEq)]
pub enum VecOrientation {
    Column,
    Row,
}

#[derive(Debug, PartialEq)]
pub enum NodeType {
    None,
    Scalar,
    Vector(VecOrientation),
    Matrix,
}
