use crate::nnetwork::CalcNode;

/// Takes the prediction as one argument and the truth as the other, and calcualted a number representing the loss. The lower the loss, the better.
pub type LossFuncType = dyn Fn(&CalcNode, &CalcNode) -> CalcNode;

/// Calculates the sum of the squares of the diviations from the truth.
pub fn least_squares(inp: &CalcNode, truth: &CalcNode) -> CalcNode {
    (inp - truth).pow(&CalcNode::new_scalar(2.)).sum()
}

/// Assumes the input can be treated as a probability distribution and that the truth is a one-hot vector
pub fn neg_log_likelihood(inp: &CalcNode, truth: &CalcNode) -> CalcNode {
    -(inp.element_wise_mul(truth)).sum().log()
}
