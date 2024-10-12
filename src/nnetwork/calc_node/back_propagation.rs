use std::{cell::RefCell, collections::HashSet};

use super::{CalcNode, CalcNodeCore, FloatType};

impl CalcNode {
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
                    if let Some(parents) = &node.borrow()._parent_nodes {
                        parents.iter().for_each(|parent| {
                            topo_sort_recursive(parent, visited, out);
                        });
                    }
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
        let sorted = topo_sort(self);
        // Initialise all gradients to zero
        sorted
            .iter()
            .for_each(|node| node.borrow_mut()._grad.iter_mut().for_each(|g| *g = 0.));
        // Initialise the root gradient to unity
        self.borrow_mut()._grad.iter_mut().for_each(|g| *g = 1.);
        // Back propagate all other gradients
        sorted.iter().rev().for_each(|node| {
            // The original nodes will not have a differentiation function
            if let Some(f) = &node.borrow()._back_propagation {
                (f)(node.clone())
            }
        });
    }

    // Adjusts values based on a constant learning rate and a previously calculated gradient
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
