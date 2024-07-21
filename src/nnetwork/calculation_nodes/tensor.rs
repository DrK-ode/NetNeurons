use std::{f64::NAN, fmt::Display};

use super::*;

impl Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field(
                "_parent_op",
                &(match self._parent_op.as_ref() {
                    Some(op) => format!("Some({})", op._op.symbol()),
                    None => "None".to_string(),
                })
                .to_string(),
            )
            .field(
                "_child_op",
                &(match self._child_op.as_ref() {
                    Some(op) => format!("Some({})", op._op.symbol()),
                    None => "None".to_string(),
                })
                .to_string(),
            )
            .field("_shape", &self._shape)
            .field("_values", &self._value)
            .field("_derivative", &self._derivative)
            .finish()
    }
}

// Access methods
impl Tensor {
    pub fn parent_op(&self) -> Option<OpNodeShared> {
        self._parent_op.clone()
    }

    pub fn child_op(&self) -> Option<OpNodeShared> {
        self._child_op.clone()
    }

    pub fn value(&self) -> &[FloatType] {
        &self._value
    }

    pub fn derivative(&self) -> &[FloatType] {
        &self._derivative
    }

    pub fn value_indexed(&self, row: usize, col: usize, depth: usize) -> Option<FloatType> {
        let index = depth * self._shape.0 * self._shape.1 + col * self._shape.0 + row;
        self._value.get(index).copied()
    }

    pub fn derivative_indexed(&self, x: usize, y: usize, z: usize) -> Option<FloatType> {
        let index = z * self._shape.0 * self._shape.1 + y * self._shape.0 + x;
        self._derivative.get(index).copied()
    }

    pub fn value_as_scalar(&self) -> Result<FloatType, TensorConversionError>{
        self.data_as_scalar(&self._value)
    }
    
    pub fn derivative_as_scalar(&self) -> Result<FloatType, TensorConversionError>{
        self.data_as_scalar(&self._derivative)
    }
    
    fn data_as_scalar(&self, data: &[FloatType] ) -> Result<FloatType, TensorConversionError> {
        if self.tensor_type() != TensorType::Scalar {
            Err(TensorConversionError {
                was_type: self.tensor_type(),
                into_type: TensorType::Scalar,
            })
        } else {
            Ok(data[0])
        }
    }

    pub fn value_as_row_vector(&self) -> Result<Vec<Vec<FloatType>>, TensorConversionError> {
        self.data_as_row_vector(&self._value)
    }
    
    pub fn derivative_as_row_vector(&self) -> Result<Vec<Vec<FloatType>>, TensorConversionError> {
        self.data_as_row_vector(&self._derivative)
    }
    
    fn data_as_row_vector(&self, data: &[FloatType]) -> Result<Vec<Vec<FloatType>>, TensorConversionError> {
        if self.tensor_type() != TensorType::Vector(VecOrientation::Row) {
            Err(TensorConversionError {
                was_type: self.tensor_type(),
                into_type: TensorType::Vector(VecOrientation::Row),
            })
        } else {
            let mut out = Vec::new();
            data.iter().for_each(|&f| out.push(vec![f]));
            Ok(out)
        }
    }

    pub fn value_as_col_vector(&self) -> Result<Vec<FloatType>, TensorConversionError> {
        self.data_as_col_vector(&self._value)
    }
    
    pub fn derivative_as_col_vector(&self) -> Result<Vec<FloatType>, TensorConversionError> {
        self.data_as_col_vector(&self._derivative)
    }
    
    fn data_as_col_vector(&self, data: &[FloatType]) -> Result<Vec<FloatType>, TensorConversionError> {
        if self.tensor_type() != TensorType::Vector(VecOrientation::Column) {
            Err(TensorConversionError {
                was_type: self.tensor_type(),
                into_type: TensorType::Vector(VecOrientation::Column),
            })
        } else {
            Ok(data.to_vec())
        }
    }

    pub fn value_as_matrix(&self) -> Result<Vec<Vec<FloatType>>, TensorConversionError> {
        self.data_as_matrix(&self._value)
    }
    
    pub fn derivative_as_matrix(&self) -> Result<Vec<Vec<FloatType>>, TensorConversionError> {
        self.data_as_matrix(&self._derivative)
    }
    
    fn data_as_matrix(&self, data: &[FloatType]) -> Result<Vec<Vec<FloatType>>, TensorConversionError> {
        if self.tensor_type() != TensorType::Matrix {
            Err(TensorConversionError {
                was_type: self.tensor_type(),
                into_type: TensorType::Matrix,
            })
        } else {
            let mut out = Vec::new();
            for i in 0..self._shape.0 {
                let begin = i * self._shape.1;
                let end = (i + 1) * self._shape.1;
                out.push(data[begin..end].to_owned());
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

impl Display for TensorType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self {
            TensorType::None => write!(f, "None"),
            TensorType::Scalar => write!(f, "Scalar"),
            TensorType::Vector(orient) => write!(
                f,
                "{} vector",
                match orient {
                    VecOrientation::Row => "Row",
                    VecOrientation::Column => "Column",
                }
            ),
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

impl Display for TensorConversionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Tried to interpret Tensor as {} but failed because it was {}",
            self.into_type, self.was_type
        )
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
            _value: vec![value],
            _derivative: vec![NAN],
            ..Default::default()
        }))
    }

    pub fn from_vector(value: Vec<FloatType>, shape: TensorShape) -> TensorShared {
        let size = value.len();
        if size != shape.0 * shape.1 * shape.2 {
            panic!("Value vector does not match the supplied shape.");
        }
        Rc::new(RefCell::new(Tensor {
            _shape: shape,
            _value: value,
            _derivative: vec![NAN; size],
            ..Default::default()
        }))
    }

    pub fn from_shape(shape: TensorShape) -> TensorShared {
        let size = shape.0 * shape.1 * shape.2;
        Rc::new(RefCell::new(Tensor {
            _shape: shape,
            _value: vec![NAN; size],
            _derivative: vec![NAN; size],
            ..Default::default()
        }))
    }

    pub fn reshape(&self, shape: TensorShape) -> TensorShared {
        assert_eq!(
            self._value.len(),
            shape.0 * shape.1 * shape.2,
            "Size must not change when reshaping Tensor"
        );
        Rc::new(RefCell::new(Tensor {
            _shape: shape,
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
                write!(f, "Scalar: [value: {}, ∇: {}]", self._value[0], self._derivative[0])?
            }
            TensorType::Vector(orient) => {
                // Column Vector
                write!(
                    f,
                    "Vector ({}): [value: {:?} ∇: {:?}]",
                    match orient {
                        VecOrientation::Column => "col",
                        VecOrientation::Row => "row",
                    },
                    self._value, self._derivative
                )?
            }
            TensorType::Matrix => {
                // Matrix
                let (x, y, _) = self._shape;
                write!(f, "Matrix ({}x{}): [value: {:?}, ∇: {:?}]", x, y, self._value, self._derivative)?
            }
            TensorType::Tensor => {
                // Some general tensor
                write!(f, "Tensor {:?}: [value: {:?}, ∇: {:?}]", self._shape, self._value, self._derivative)?
            }
        }
        Ok(())
    }
}
