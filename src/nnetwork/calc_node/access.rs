use std::{cell::RefCell, fmt::Display, ops::Deref, rc::Rc};

use super::{CalcNode, CalcNodeCore, FloatType, NodeShape, NodeType, VecOrientation};

impl CalcNodeCore {
    pub fn vals(&self) -> &[FloatType] {
        &self._vals
    }
    pub fn grad(&self) -> &[FloatType] {
        &self._grad
    }
    pub fn parents(&self) -> &[CalcNode]{
        &self._parent_nodes
    }
    pub fn back_propagation(&self) -> &Option<Box<dyn Fn(CalcNode)>>{
        &self._back_propagation
    }
    pub fn shape(&self) -> &NodeShape {
        &self._shape
    }
}

impl Deref for CalcNode {
    type Target = Rc<RefCell<CalcNodeCore>>;

    fn deref(&self) -> &Self::Target {
        &self._node
    }
}

impl CalcNode {
    pub fn len(&self) -> usize {
        {
            let shape: &NodeShape = &self.shape();
            shape.0 * shape.1
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl CalcNode {
    pub fn node_type(&self) -> NodeType {
        match self.borrow()._shape {
            (0, _) | (_, 0) => {
                // No value
                NodeType::None
            }
            (1, 1) => {
                // Scalar
                NodeType::Scalar
            }
            (_, 1) => {
                // Column Vector
                NodeType::Vector(VecOrientation::Column)
            }
            (1, _) => {
                // Row Vector
                NodeType::Vector(VecOrientation::Row)
            }
            (_, _) => {
                // Matrix
                NodeType::Matrix
            }
        }
    }
}

impl Display for CalcNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.node_type() {
            NodeType::None => {
                // No value
                write!(f, "None")?
            }
            NodeType::Scalar => {
                // Scalar
                write!(
                    f,
                    "Scalar: [value: {}, ∇: {}]",
                    self.borrow()._vals[0],
                    self.borrow()._grad[0]
                )?
            }
            NodeType::Vector(orient) => {
                // Column Vector
                write!(
                    f,
                    "Vector ({}: {}): [value: {:?} ∇: {:?}]",
                    match orient {
                        VecOrientation::Column => "cols",
                        VecOrientation::Row => "rows",
                    },
                    self.len(),
                    self.borrow()._vals,
                    self.borrow()._grad
                )?
            }
            NodeType::Matrix => {
                // Matrix
                let (x, y) = self.borrow()._shape;
                write!(
                    f,
                    "Matrix ({}x{}): [value: {:?}, ∇: {:?}]",
                    x,
                    y,
                    self.borrow()._vals,
                    self.borrow()._grad
                )?
            }
        }
        Ok(())
    }
}

impl Display for NodeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self {
            NodeType::None => write!(f, "None"),
            NodeType::Scalar => write!(f, "Scalar"),
            NodeType::Vector(orient) => write!(
                f,
                "{} vector",
                match orient {
                    VecOrientation::Row => "Row",
                    VecOrientation::Column => "Column",
                }
            ),
            NodeType::Matrix => write!(f, "Matrix"),
        }
    }
}

// Access methods
impl CalcNode {
    pub fn copy_parents(&self) -> Vec<CalcNode> {
        self.borrow()._parent_nodes.clone()
    }

    pub fn shape(&self) -> NodeShape {
        self.borrow()._shape
    }

    pub fn copy_vals(&self) -> Vec<FloatType> {
        self.borrow()._vals.clone()
    }

    pub fn copy_grad(&self) -> Vec<FloatType> {
        self.borrow()._grad.clone()
    }

    pub fn value_indexed(&self, i: usize) -> FloatType {
        self.borrow()._vals[i]
    }

    pub fn gradient_indexed(&self, i: usize) -> FloatType {
        self.borrow()._grad[i]
    }

    pub fn set_vals(&mut self, vals: &[FloatType]) {
        assert_eq!(vals.len(), self.borrow()._vals.len());
        self.borrow_mut()._vals = vals.to_vec();
    }

    pub fn add_grad(&mut self, grad: &[FloatType]) {
        assert_eq!(grad.len(), self.borrow()._grad.len());
        self.borrow_mut()._grad.iter_mut().zip(grad.iter()).for_each(|(target, &value)| {
            *target += value});
    }
    
    pub fn set_grad(&mut self, grad: &[FloatType]) {
        assert_eq!(grad.len(), self.borrow()._grad.len());
        self.borrow_mut()._grad = grad.to_vec();
    }
    
    pub fn set_value_indexed(&mut self, i:usize, val: FloatType) {
        assert!( i < self.len());
        self.borrow_mut()._vals[i] = val;
    }
    
    pub fn set_gradient_indexed(&mut self, i:usize, val: FloatType) {
        assert!( i < self.len());
        self.borrow_mut()._grad[i] = val;
    }
}
