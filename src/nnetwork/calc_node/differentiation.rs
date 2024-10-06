use std::{cell::RefCell, collections::HashSet};

use super::{CalcNode, CalcNodeShared, FloatType};

impl CalcNodeShared{
    pub fn back_propagation(&self) {
        // Returns a sorted list of CalcNodeShared
        fn topo_sort(root: &CalcNodeShared) -> Vec<CalcNodeShared> {
            // This is were the magic happens
            fn topo_sort_recursive(
                node: &CalcNodeShared,
                visited: &mut HashSet<usize>,
                out: &mut Vec<CalcNodeShared>,
            ) {
                fn ptr_as_usize(node: &CalcNodeShared) -> usize {
                    (node.as_ptr() as *const CalcNode) as usize
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
            let mut sorted: Vec<CalcNodeShared> = Vec::new();
            // Finds all parents (and their parents) and adds them to the vector before adding the root
            topo_sort_recursive(root, &mut visited, &mut sorted);
            sorted
        }
        // The end result will be at the end of the vector
        let sorted = topo_sort(self);
        // Initialise all gradients to zero
        sorted
            .iter()
            .for_each(|node| node.borrow_mut()._grad.iter_mut().for_each(|g| *g = 0.));
        // Initialise the root gradient to unity
        self.borrow_mut()._grad.iter_mut().for_each(|g| *g = 1.);
        // Calculate all other gradients backwards
        sorted.iter().rev().for_each(|node| {
            // The original nodes will not have a differentiation function
            if let Some(f) = &node.borrow()._back_propagation {
                (f)(node.clone())
            }
        });
    }
    
    pub fn decend_grad(&self, learning_rate: FloatType){
        let mut tmp = RefCell::new(CalcNode::default());
        self.swap(&tmp);
        let t = tmp.get_mut();
        t._vals
            .iter_mut()
            .zip(t._grad.iter())
            .for_each(|(v, d)| *v -= learning_rate * d);
        self.swap(&tmp);
    }
}