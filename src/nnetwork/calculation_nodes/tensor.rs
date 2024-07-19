use std::{f64::NAN, fmt::Display};

use super::*;

// Access methods
impl Tensor {
    pub fn parent_op(&self) -> Option<OpNodeShared> {
        self._parent_op.clone()
    }

    pub fn child_op(&self) -> Option<OpNodeShared> {
        self._child_op.clone()
    }

    pub fn value(&self, x: usize, y: usize, z: usize) -> Option<FloatType> {
        let index = z * self._shape.0 * self._shape.1 + y * self._shape.0 + x;
        self._values.get(index).copied()
    }

    pub fn derivative(&self, x: usize, y: usize, z: usize) -> Option<FloatType> {
        let index = z * self._shape.0 * self._shape.1 + y * self._shape.0 + x;
        self._derivative.get(index).copied()
    }

    pub fn as_scalar(&self) -> Result<FloatType, TensorConversionError> {
        if self.tensor_type() != TensorType::Scalar {
            Err(TensorConversionError {
                was_type: self.tensor_type(),
                into_type: TensorType::Scalar,
            })
        } else {
            Ok(self._values[0])
        }
    }

    pub fn as_col_vector(&self) -> Result<Vec<Vec<FloatType>>, TensorConversionError> {
        if self.tensor_type() != TensorType::Vector(VecOrientation::Column) {
            Err(TensorConversionError {
                was_type: self.tensor_type(),
                into_type: TensorType::Vector(VecOrientation::Column),
            })
        } else {
            let mut out = Vec::new();
            self._values.iter().for_each(|&f| out.push(vec![f]));
            Ok(out)
        }
    }

    pub fn as_row_vector(&self) -> Result<Vec<FloatType>, TensorConversionError> {
        if self.tensor_type() != TensorType::Vector(VecOrientation::Row) {
            Err(TensorConversionError {
                was_type: self.tensor_type(),
                into_type: TensorType::Vector(VecOrientation::Row),
            })
        } else {
            Ok(self._values.clone())
        }
    }

    pub fn as_matrix(&self) -> Result<Vec<Vec<FloatType>>, TensorConversionError> {
        if self.tensor_type() != TensorType::Matrix {
            Err(TensorConversionError {
                was_type: self.tensor_type(),
                into_type: TensorType::Matrix,
            })
        } else {
            let mut out = Vec::new();
            for i in 0..self._shape.0 {
                let begin = i*self._shape.1;
                let end = (i+1)*self._shape.1;
                out.push( self._values[begin..end].to_owned() );
            }
            Ok(out)
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum VecOrientation {
    Column,
    Row,
}

#[derive(Debug, PartialEq)]
pub enum TensorType {
    None,
    Scalar,
    Vector(VecOrientation),
    Matrix,
    Tensor,
}

impl Display for TensorType{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self{
            TensorType::None =>write!(f,  "None"),
                TensorType::Scalar => write!(f, "Scalar"),
            TensorType::Vector(orient) => write!(f, "{} vector", match orient {
                VecOrientation::Row => "Row",
                VecOrientation::Column => "Column",
            }),
            TensorType::Matrix => write!(f, "Matrix"),
            TensorType::Tensor => write!(f, "Tensor"),
        }
    }
}

#[derive(Debug)]
pub struct TensorConversionError {
    was_type: TensorType,
    into_type: TensorType,
}

impl Display for TensorConversionError{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f,"Tried to interpret Tensor as {} but failed because it was {}", self.into_type, self.was_type)
    }
}

// Tensor properties
impl Tensor {
    pub fn tensor_type(&self) -> TensorType {
        match self._shape {
            (0, 0, 0) => {
                // No value
                TensorType::None
            }
            (1, 1, 1) => {
                // Scalar
                TensorType::Scalar
            }
            (x, 1, 1) if x > 1 => {
                // Column Vector
                TensorType::Vector(VecOrientation::Column)
            }
            (1, y, 1) if y > 1 => {
                // Row Vector
                TensorType::Vector(VecOrientation::Row)
            }
            (x, y, 1) if x > 1 && y > 1 => {
                // Matrix
                TensorType::Matrix
            }
            _ => {
                // Some general tensor
                TensorType::Tensor
            }
        }
    }
}

// Ctors
impl Tensor {
    pub fn new() -> TensorShared {
        Rc::new(RefCell::new(Tensor::default()))
    }

    pub fn from_scalar(value: FloatType) -> TensorShared {
        Rc::new(RefCell::new(Tensor {
            _shape: (1, 1, 1),
            _values: vec![value],
            ..Default::default()
        }))
    }

    pub fn from_vector(value: Vec<FloatType>, shape: (usize, usize, usize)) -> TensorShared {
        if value.len() != shape.0 * shape.1 * shape.2 {
            panic!("Value vector does not match the supplied shape.");
        }
        Rc::new(RefCell::new(Tensor {
            _shape: shape,
            _values: value,
            ..Default::default()
        }))
    }
    
    pub fn from_shape(shape: (usize, usize, usize)) -> TensorShared {
        let size =  shape.0 * shape.1 * shape.2;
        Rc::new(RefCell::new(Tensor {
            _shape: shape,
            _values: vec![NAN;size],
            ..Default::default()
        }))
    }

    pub fn reshape(tensor: Tensor, shape: (usize, usize, usize)) -> TensorShared {
        let mut new_value = tensor._values;
        new_value.resize(shape.0 * shape.1 * shape.2, 0.);
        Rc::new(RefCell::new(Tensor {
            _shape: shape,
            _values: new_value,
            ..Default::default()
        }))
    }
}

impl Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.tensor_type() {
            TensorType::None => {
                // No value
                write!(f, "None")?
            }
            TensorType::Scalar => {
                // Scalar
                write!(f, "Scalar: {}", self._values[0])?
            }
            TensorType::Vector(orient) => {
                // Column Vector
                write!(
                    f,
                    "Vector ({}): {:?}",
                    match orient {
                        VecOrientation::Column => "col",
                        VecOrientation::Row => "row",
                    },
                    self._values
                )?
            }
            TensorType::Matrix => {
                // Matrix
                let (x, y, _) = self._shape;
                write!(f, "Matrix ({}x{}): {:?}", x, y, self._values)?
            }
            TensorType::Tensor => {
                // Some general tensor
                write!(f, "Tensor {:?}: {:?}", self._shape, self._values)?
            }
        }
        Ok(())
    }
}
