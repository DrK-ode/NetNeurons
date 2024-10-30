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

    /// Returns a new instance of a [CalcNode] containing a [CalcNodeCore] with the supplied member data. Will panic if the size of the [NodeShape] and data are not equal.
    ///
    /// # Example 1
    /// ```
    /// use net_neurons::nnetwork::CalcNode;
    /// 
    /// let shape = (1,1);
    /// let vals = vec![1.];
    /// let parents = vec![];
    /// let node1 = CalcNode::new(shape, vals, parents, None);
    ///
    /// let shape = (1,1);
    /// let vals = vec![1.];
    /// let parents = vec![node1];
    /// let node2 = CalcNode::new(
    ///     shape,
    ///     vals,
    ///     parents,
    ///     Some( Box::new( |child_node: CalcNode| {
    ///         for mut p in child_node.copy_parents(){
    ///             p.add_grad(&vec![1.]);
    ///         }
    ///     })));
    /// ```
    pub fn new(
        shape: NodeShape,
        vals: Vec<FloatType>,
        parents: Vec<CalcNode>,
        back_propagation: Option<Box<dyn Fn(CalcNode)>>,
    ) -> Self {
        let size = Self::size_of_shape(&shape);
        assert_eq!(size, vals.len());
        CalcNode {
            _node: Rc::new(RefCell::new(CalcNodeCore {
                _shape: shape,
                _vals: vals,
                _grad: vec![FloatType::NAN; size],
                _parent_nodes: parents,
                _back_propagation: back_propagation,
            })),
        }
    }

    /// Shortcut for constructing scalar nodes.
    pub fn new_scalar(value: FloatType) -> Self {
        Self::new_from_shape((1, 1), vec![value])
    }

    /// Shortcut for constructing column vector nodes.
    pub fn new_col_vector(value: Vec<FloatType>) -> Self {
        let size = value.len();
        Self::new_from_shape((size, 1), value)
    }

    /// Shortcut for constructing row vector nodes.
    pub fn new_row_vector(value: Vec<FloatType>) -> Self {
        let size = value.len();
        Self::new_from_shape((1, size), value)
    }

    /// Constructs a node with a specific shape and values. Will panic if the size of the shape and data are not equal.
    ///
    /// # Example
    /// ```
    /// use net_neurons::nnetwork::CalcNode;
    /// 
    /// let shape = (2,2); // a 2 by 2 matrix
    /// let vals = vec![1.,2.,3.,4.];
    /// let node = CalcNode::new_from_shape(shape, vals);
    /// ```
    pub fn new_from_shape(shape: NodeShape, vals: Vec<FloatType>) -> Self {
        Self::new(shape, vals, vec![], None)
    }

    /// Similar to [CalcNode::new_from_shape] but randomizes all values from a normal distribution.
    pub fn rand_from_shape(shape: NodeShape) -> Self {
        let size = Self::size_of_shape(&shape);
        Self::new_from_shape(
            shape,
            (0..size)
                .map(|_| thread_rng().sample(StandardNormal))
                .collect(),
        )
    }

    /// Coerce the node into a new shape. Will panic if the size changes.
    pub fn reshape(&mut self, shape: NodeShape) {
        assert_eq!(
            self._node.borrow()._vals.len(),
            Self::size_of_shape(&shape),
            "Size must not change when reshaping Tensor"
        );
        self.borrow_mut()._shape = shape;
    }
}
