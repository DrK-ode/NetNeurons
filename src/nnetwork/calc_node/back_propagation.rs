use std::{cell::RefCell, collections::HashSet};

use super::{CalcNode, CalcNodeCore, FloatType};

impl CalcNode {
    /// Recalculates the gradients of all nodes leading up to the current one. The gradient of the current node is set to unity.
    /// 
    /// # Example
    /// ```
    /// use net_neurons::nnetwork::CalcNode;
    /// 
    /// let a = CalcNode::new_scalar(2.);
    /// let mut b = &a * &a;
    /// b.back_propagation();
    /// assert_eq!(a.gradient_indexed(0), 4.);
    /// assert_eq!(b.gradient_indexed(0), 1.);
    /// ```
    pub fn back_propagation(&mut self) {
        // Returns a sorted list of CalcNodes
        fn topo_sort(root: &CalcNode) -> Vec<CalcNode> {
            // Recursive function that does the actual sorting
            fn topo_sort_recursive(
                node: &CalcNode,
                visited: &mut HashSet<usize>,
                out: &mut Vec<CalcNode>,
            ) {
                fn ptr_as_usize(node: &CalcNode) -> usize {
                    (node.as_ptr() as *const CalcNodeCore) as usize
                }
                if !visited.contains(&ptr_as_usize(node)) {
                    visited.insert(ptr_as_usize(node));
                    node.borrow()._parent_nodes.iter().for_each(|parent| {
                        topo_sort_recursive(parent, visited, out);
                    });
                    out.push(node.clone());
                }
            }
            // These container will be sent down the recursive calls
            let mut visited: HashSet<usize> = HashSet::new();
            let mut sorted: Vec<CalcNode> = Vec::new();
            // Finds all parents (and their parents) and adds them to the vector before adding the root
            topo_sort_recursive(root, &mut visited, &mut sorted);
            sorted
        }

        // The final result will be at the end of the vector
        let mut sorted = topo_sort(self);
        // Initialise all gradients to zero
        sorted
            .iter_mut()
            .for_each(|node| node.reset_grad());
        // Initialise the root gradient to unity
        self.set_grad(&vec![1.;self.len()]);
        // Back propagate all other gradients
        sorted.iter().rev().for_each(|node| {
            // The original nodes will not have a differentiation function
            if let Some(f) = &node.borrow()._back_propagation {
                (f)(node.clone())
            }
        });
    }

    /// Decends the gradient by a fraction of the calculated gradient.
    pub fn decend_grad(&mut self, learning_rate: FloatType) {
        let mut tmp = RefCell::new(CalcNodeCore::default());
        // Bring the node outside the RefCell since we need to borrow both values and gradients at the same time
        self.swap(&tmp);
        let t = tmp.get_mut();
        t._vals
            .iter_mut()
            .zip(t._grad.iter())
            .for_each(|(v, d)| *v -= learning_rate * d);
        self.swap(&tmp);
    }
}
