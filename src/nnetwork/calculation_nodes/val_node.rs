use std::fmt::Display;

use super::*;

// Access methods
impl ValNode {
    pub fn parent_op(&self) -> Option<OpNodeShared> {
        self._parent_op.clone()
    }
    pub fn child_op(&self) -> Option<OpNodeShared> {
        self._child_op.clone()
    }
    pub fn value(&self) -> Option<NodeValue> {
        self._value.clone()
    }
    pub fn derivative(&self) -> Option<NodeValue> {
        self._derivative.clone()
    }
}

impl Display for NodeValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NodeValue::Scalar(a) => write!(f, "Scalar ({a})"),
            NodeValue::Vector(v) => {
                write!(f, "Vector [");
                v.iter().for_each(|a| {
                    write!(f, "{a}, ");
                });
                write!(f, "]")
            }
        }
    }
}

impl Display for ValNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Value: ");
        if self._value.is_some() {
            write!(f, "{}", self._value.as_ref().unwrap())
        } else {
            write!(f, "None")
        };
        if self._derivative.is_some() {
            write!(f, ", : {}", &self._derivative.as_ref().unwrap());
        }
        Ok(())
    }
}
