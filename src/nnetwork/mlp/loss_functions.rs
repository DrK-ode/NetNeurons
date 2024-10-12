use crate::nnetwork::CalcNode;

pub fn least_squares(inp: &CalcNode, truth: &CalcNode) -> CalcNode {
    (inp - truth).pow(&CalcNode::new_scalar(2.)).sum()
}

// Assumes the input can be treated as a probability distribution and that the truth is a one-hot vector
pub fn neg_log_likelihood(inp: &CalcNode, truth: &CalcNode) -> CalcNode {
    -(inp.element_wise_mul(truth)).sum().log()
}
