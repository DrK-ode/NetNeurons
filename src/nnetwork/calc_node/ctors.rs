use std::{cell::RefCell, rc::Rc};

use rand::{thread_rng, Rng};
use rand_distr::StandardNormal;

use super::{
    types::{FloatType, NodeShape},
    CalcNode, CalcNodeCore,
};

// Ctors
impl CalcNode {
    fn size_of_shape(shape: &NodeShape) -> usize {
        shape.0 * shape.1
    }

    pub fn new(
        shape: NodeShape,
        vals: Vec<FloatType>,
        parents: Vec<CalcNode>,
        back_propagation: Option<Box<dyn Fn(CalcNode)>>,
    ) -> Self {
        let size = Self::size_of_shape(&shape);
        assert_eq!(size, vals.len());
        Self::from_node(CalcNodeCore {
            _shape: shape,
            _vals: vals,
            _grad: vec![f64::NAN; size],
            _parent_nodes: parents,
            _back_propagation: back_propagation
        })
    }

    fn from_node(n: CalcNodeCore) -> Self {
        Self {
            _node: Rc::new(RefCell::new(n)),
        }
    }

    pub fn new_scalar(value: FloatType) -> Self {
        Self::filled_from_shape((1, 1), vec![value])
    }

    pub fn new_col_vector(value: Vec<FloatType>) -> Self {
        let size = value.len();
        Self::filled_from_shape((size, 1), value)
    }

    pub fn new_row_vector(value: Vec<FloatType>) -> Self {
        let size = value.len();
        Self::filled_from_shape((1, size), value)
    }

    pub fn empty_from_shape(shape: NodeShape) -> Self {
        let size = Self::size_of_shape(&shape);
        Self::from_node(CalcNodeCore {
            _shape: shape,
            _vals: vec![f64::NAN; size],
            _grad: vec![f64::NAN; size],
            ..Default::default()
        })
    }

    pub fn filled_from_shape(shape: NodeShape, vals: Vec<FloatType>) -> Self {
        Self::new(shape, vals, vec![], None)
    }

    pub fn rand_from_shape(shape: NodeShape) -> Self {
        let size = Self::size_of_shape(&shape);
        Self::from_node(CalcNodeCore {
            _shape: shape,
            _vals: (0..size)
                .map(|_| thread_rng().sample(StandardNormal))
                .collect(),
            _grad: vec![f64::NAN; size],
            ..Default::default()
        })
    }

    pub fn reshape(&mut self, shape: NodeShape) {
        assert_eq!(
            self._node.borrow()._vals.len(),
            Self::size_of_shape(&shape),
            "Size must not change when reshaping Tensor"
        );
        self.borrow_mut()._shape = shape;
    }
}
