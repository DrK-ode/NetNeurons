use std::{
    fmt::Display,
    iter::Sum,
    ops::{Add, Div, Mul, Neg, Sub},
};

use op_node::{AddOp, DotOp, ExpOp, LogOp, MulOp, NegOp, PowOp, ProdOp, SumOp};
use rand::{thread_rng, Rng};
use rand_distr::StandardNormal;

use super::*;

#[derive(Debug)]
pub struct TensorConversionError {
    was_type: TensorType,
    into_type: TensorType,
}

impl Deref for TensorShared {
    type Target = Rc<RefCell<Tensor>>;

    fn deref(&self) -> &Self::Target {
        &self._tensor
    }
}

// Ctors
impl TensorShared {
    pub fn new() -> Self {
        Self::from_tensor(Tensor::default())
    }

    pub fn from_tensor(t: Tensor) -> Self {
        Self {
            _tensor: Rc::new(RefCell::new(t)),
        }
    }

    pub fn from_scalar(value: FloatType) -> Self {
        Self::from_tensor(Tensor {
            _shape: (1, 1, 1),
            _value: vec![value],
            _derivative: vec![f64::NAN],
            ..Default::default()
        })
    }

    pub fn from_vector(value: Vec<FloatType>, shape: TensorShape) -> Self {
        let size = value.len();
        if size != shape.0 * shape.1 * shape.2 {
            panic!("Value vector does not match the supplied shape.");
        }
        Self::from_tensor(Tensor {
            _shape: shape,
            _value: value,
            _derivative: vec![f64::NAN; size],
            ..Default::default()
        })
    }

    pub fn from_shape(shape: TensorShape) -> Self {
        let size = shape.0 * shape.1 * shape.2;
        Self::from_tensor(Tensor {
            _shape: shape,
            _value: vec![f64::NAN; size],
            _derivative: vec![f64::NAN; size],
            ..Default::default()
        })
    }

    pub fn from_random(shape: TensorShape) -> Self {
        let size = shape.0 * shape.1 * shape.2;
        Self::from_tensor(Tensor {
            _shape: shape,
            _value: (0..size)
                .map(|_| thread_rng().sample(StandardNormal))
                .collect(),
            _derivative: vec![f64::NAN; size],
            ..Default::default()
        })
    }

    pub fn reshape(&mut self, shape: TensorShape) {
        assert_eq!(
            self._tensor.borrow()._value.len(),
            shape.0 * shape.1 * shape.2,
            "Size must not change when reshaping Tensor"
        );
        self.borrow_mut()._shape = shape;
    }
}

impl From<&[FloatType]> for TensorShared {
    fn from(value: &[FloatType]) -> Self {
        TensorShared::from_vector(value.to_vec(), (value.len(), 1, 1))
    }
}
impl From<Vec<FloatType>> for TensorShared {
    fn from(value: Vec<FloatType>) -> Self {
        let size = value.len();
        TensorShared::from_vector(value, (size, 1, 1))
    }
}

impl TensorShared {
    pub fn decend_grad(&self, learning_rate: FloatType) {
        let mut tmp = RefCell::new(Tensor::default());
        self.swap(&tmp);
        let t = tmp.get_mut();
        t._value
            .iter_mut()
            .zip(t._derivative.iter())
            .for_each(|(v, d)| *v -= learning_rate * d);
        self.swap(&tmp);
    }
}

impl Tensor {
    pub fn value(&self) -> &[FloatType] {
        &self._value
    }

    pub fn derivative(&self) -> &[FloatType] {
        &self._derivative
    }
}

// Access methods
impl TensorShared {
    pub fn parent_op(&self) -> Option<OpNodeShared> {
        self.borrow()._parent_op.clone()
    }

    pub fn child_op(&self) -> Option<OpNodeShared> {
        self.borrow()._child_op.clone()
    }

    pub fn value(&self) -> Vec<FloatType> {
        self.borrow()._value.clone()
    }

    pub fn derivative(&self) -> Vec<FloatType> {
        self.borrow()._derivative.clone()
    }

    pub fn value_indexed(&self, row: usize, col: usize, depth: usize) -> Option<FloatType> {
        let shape = self.borrow()._shape;
        let index = depth * shape.0 * shape.1 + col * shape.0 + row;
        self.borrow()._value.get(index).copied()
    }

    pub fn derivative_indexed(&self, x: usize, y: usize, z: usize) -> Option<FloatType> {
        let shape = self.borrow()._shape;
        let index = z * shape.0 * shape.1 + y * shape.0 + x;
        self.borrow()._derivative.get(index).copied()
    }

    pub fn value_as_scalar(&self) -> Result<FloatType, TensorConversionError> {
        self.data_as_scalar(&self.borrow()._value)
    }

    pub fn derivative_as_scalar(&self) -> Result<FloatType, TensorConversionError> {
        self.data_as_scalar(&self.borrow()._derivative)
    }

    fn data_as_scalar(&self, data: &[FloatType]) -> Result<FloatType, TensorConversionError> {
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
        self.data_as_row_vector(&self.borrow()._value)
    }

    pub fn derivative_as_row_vector(&self) -> Result<Vec<Vec<FloatType>>, TensorConversionError> {
        self.data_as_row_vector(&self.borrow()._derivative)
    }

    fn data_as_row_vector(
        &self,
        data: &[FloatType],
    ) -> Result<Vec<Vec<FloatType>>, TensorConversionError> {
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
        self.data_as_col_vector(&self.borrow()._value)
    }

    pub fn derivative_as_col_vector(&self) -> Result<Vec<FloatType>, TensorConversionError> {
        self.data_as_col_vector(&self.borrow()._derivative)
    }

    fn data_as_col_vector(
        &self,
        data: &[FloatType],
    ) -> Result<Vec<FloatType>, TensorConversionError> {
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
        self.data_as_matrix(&self.borrow()._value)
    }

    pub fn derivative_as_matrix(&self) -> Result<Vec<Vec<FloatType>>, TensorConversionError> {
        self.data_as_matrix(&self.borrow()._derivative)
    }

    fn data_as_matrix(
        &self,
        data: &[FloatType],
    ) -> Result<Vec<Vec<FloatType>>, TensorConversionError> {
        if self.tensor_type() != TensorType::Matrix {
            Err(TensorConversionError {
                was_type: self.tensor_type(),
                into_type: TensorType::Matrix,
            })
        } else {
            let (n_rows, n_cols, _) = self.borrow()._shape;
            let mut out = Vec::new();
            for row in 0..n_rows {
                out.push(
                    data.iter()
                        .skip(row * n_cols)
                        .take(n_cols)
                        .copied()
                        .collect(),
                );
            }
            Ok(out)
        }
    }
}

// Set methods
impl Tensor {
    pub fn set_value(&mut self, v: Vec<FloatType>) {
        self._value = v;
    }
}
impl TensorShared {
    pub fn set_value(&mut self, v: &[FloatType]) {
        assert_eq!(
            self.len(),
            v.len(),
            "Cannot overwrite Tensor value with value of new size."
        );
        self.borrow_mut().set_value(v.into());
    }
    
    pub fn set_index(&mut self, i:usize,j:usize,k:usize, v: FloatType){
        let (_n_i, n_j, n_k) = self.shape();
        let index = i * n_j*n_k + j*n_k + k;
        self.borrow_mut()._value[index] = v;
    }
}

// Tensor properties
impl Tensor {
    pub fn len(&self) -> usize {
        self._value.len()
    }
}
impl TensorShared {
    pub fn len(&self) -> usize {
        self.borrow().len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn shape(&self) -> TensorShape {
        self.borrow()._shape
    }
    pub fn tensor_type(&self) -> TensorType {
        match self.borrow()._shape {
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

// Operations
impl TensorShared {
    pub fn exp(&self) -> TensorShared {
        OpNode::new_op(Box::new(ExpOp {}), vec![self.clone()], true)
    }

    pub fn log(&self) -> TensorShared {
        OpNode::new_op(Box::new(LogOp {}), vec![self.clone()], true)
    }

    pub fn inv(&self) -> TensorShared {
        OpNode::new_op(
            Box::new(PowOp {}),
            vec![
                self.clone(),
                TensorShared::from_vector(vec![-1.; self.len()], self.shape()),
            ],
            true,
        )
    }

    pub fn pow(&self, rhs: &TensorShared) -> TensorShared {
        if rhs.len() == 1 {
            OpNode::new_op(Box::new(PowOp {}), vec![self.clone(), rhs.clone()], false)
        } else {
            OpNode::new_op(Box::new(PowOp {}), vec![self.clone(), rhs.clone()], true)
        }
    }

    pub fn powf(&self, rhs: FloatType) -> TensorShared {
        self.pow(&TensorShared::from_scalar(rhs))
    }

    pub fn sum(&self) -> TensorShared {
        // Sum adds all elements in a tensor together. Implemented as a variant of Add.
        OpNode::new_op(Box::new(SumOp {}), vec![self.clone()], true)
    }

    pub fn product(&self) -> TensorShared {
        // Prod multiplies all elements in a tensor together. Implemented as a unary variant of Mul.
        OpNode::new_op(Box::new(ProdOp {}), vec![self.clone()], true)
    }

    pub fn add_many(inp: &[TensorShared]) -> TensorShared {
        OpNode::new_op(Box::new(AddOp {}), inp.to_owned(), true)
    }

    pub fn mul_many(inp: &[TensorShared]) -> TensorShared {
        OpNode::new_op(Box::new(MulOp {}), inp.to_owned(), true)
    }

    pub fn dot(&self, rhs: &TensorShared) -> TensorShared {
        OpNode::new_op(Box::new(DotOp {}), vec![self.clone(), rhs.clone()], false)
    }

    pub fn normalized(&self) -> TensorShared {
        self / self.sum()
    }
}

impl Neg for TensorShared {
    type Output = TensorShared;
    fn neg(self) -> Self::Output {
        -(&self)
    }
}
impl Neg for &TensorShared {
    type Output = TensorShared;

    fn neg(self) -> Self::Output {
        OpNode::new_op(Box::new(NegOp {}), vec![self.clone()], true)
    }
}
impl Add for &TensorShared {
    type Output = TensorShared;
    fn add(self, rhs: Self) -> Self::Output {
        OpNode::new_op(Box::new(AddOp {}), vec![self.clone(), rhs.clone()], true)
    }
}
impl Add for TensorShared {
    type Output = TensorShared;
    fn add(self, rhs: Self) -> Self::Output {
        &self + &rhs
    }
}
impl Add<&TensorShared> for TensorShared {
    type Output = TensorShared;
    fn add(self, rhs: &Self) -> Self::Output {
        &self + rhs
    }
}
impl Add<TensorShared> for &TensorShared {
    type Output = TensorShared;
    fn add(self, rhs: TensorShared) -> Self::Output {
        self + &rhs
    }
}
impl Sub for TensorShared {
    type Output = TensorShared;
    fn sub(self, rhs: Self) -> Self::Output {
        &self - &rhs
    }
}
impl Sub for &TensorShared {
    type Output = TensorShared;
    fn sub(self, rhs: Self) -> Self::Output {
        OpNode::new_op(Box::new(AddOp {}), vec![self.clone(), -rhs], true)
    }
}
impl Sub<&TensorShared> for TensorShared {
    type Output = TensorShared;
    fn sub(self, rhs: &Self) -> Self::Output {
        &self - rhs
    }
}
impl Sub<TensorShared> for &TensorShared {
    type Output = TensorShared;
    fn sub(self, rhs: TensorShared) -> Self::Output {
        self - &rhs
    }
}
impl Mul for TensorShared {
    type Output = TensorShared;
    fn mul(self, rhs: Self) -> Self::Output {
        &self * &rhs
    }
}
impl Mul for &TensorShared {
    type Output = TensorShared;
    fn mul(self, rhs: Self) -> Self::Output {
        if rhs.len() == 1 {
            OpNode::new_op(Box::new(MulOp {}), vec![self.clone(), rhs.clone()], false)
        } else {
            OpNode::new_op(Box::new(MulOp {}), vec![self.clone(), rhs.clone()], true)
        }
    }
}
impl Mul<&TensorShared> for TensorShared {
    type Output = TensorShared;
    fn mul(self, rhs: &Self) -> Self::Output {
        &self * rhs
    }
}
impl Mul<TensorShared> for &TensorShared {
    type Output = TensorShared;
    fn mul(self, rhs: TensorShared) -> Self::Output {
        self * &rhs
    }
}
impl Div for TensorShared {
    type Output = TensorShared;
    fn div(self, rhs: Self) -> Self::Output {
        &self / &rhs
    }
}
#[allow(clippy::suspicious_arithmetic_impl)]
impl Div for &TensorShared {
    type Output = TensorShared;
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.inv()
    }
}
impl Div<&TensorShared> for TensorShared {
    type Output = TensorShared;
    fn div(self, rhs: &Self) -> Self::Output {
        &self / rhs
    }
}
impl Div<TensorShared> for &TensorShared {
    type Output = TensorShared;
    fn div(self, rhs: TensorShared) -> Self::Output {
        self / &rhs
    }
}
impl Sum for TensorShared {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        TensorShared::add_many(&iter.collect::<Vec<_>>())
    }
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

impl Display for TensorShared {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.tensor_type() {
            TensorType::None => {
                // No value
                write!(f, "None")?
            }
            TensorType::Scalar => {
                // Scalar
                write!(
                    f,
                    "Scalar: [value: {}, ∇: {}]",
                    self.borrow()._value[0],
                    self.borrow()._derivative[0]
                )?
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
                    self.borrow()._value,
                    self.borrow()._derivative
                )?
            }
            TensorType::Matrix => {
                // Matrix
                let (x, y, _) = self.borrow()._shape;
                write!(
                    f,
                    "Matrix ({}x{}): [value: {:?}, ∇: {:?}]",
                    x,
                    y,
                    self.borrow()._value,
                    self.borrow()._derivative
                )?
            }
            TensorType::Tensor => {
                // Some general tensor
                write!(
                    f,
                    "Tensor {:?}: [value: {:?}, ∇: {:?}]",
                    self.borrow()._shape,
                    self.borrow()._value,
                    self.borrow()._derivative
                )?
            }
        }
        Ok(())
    }
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
