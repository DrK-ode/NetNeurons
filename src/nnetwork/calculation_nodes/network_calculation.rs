use std::{collections::HashSet, time::Instant};

use super::*;

impl NetworkCalculation {
    pub fn new(root: &TensorShared) -> Self {
        NetworkCalculation {
            _op_order: Self::topo_sort(root),
        }
    }

    pub fn forward(&self) -> TensorShared {
        self._op_order.iter().for_each(|op| op.perform_operation());
        self._op_order.last().unwrap()._out.clone()
    }

    pub fn back_propagation(&self) {
        // Set dx/dx to 1 for the root node
        self._op_order
            .last()
            .unwrap()
            ._out
            .borrow_mut()
            ._derivative
            .iter_mut()
            .for_each(|d| *d = 1.);
        // Initialize d/dx to 0 for all other nodes
        self._op_order.iter().for_each(|op| {
            op._inp
                .iter()
                .for_each(|t| t.borrow_mut()._derivative.iter_mut().for_each(|d| *d = 0.))
        });
        // Calculate all other derivatives backwards
        self._op_order
            .iter()
            .rev()
            .for_each(|op| op.back_propagate());
    }
}

impl NetworkCalculation {
    fn topo_sort(root: &TensorShared) -> Vec<OpNodeShared> {
        fn topo_sort_recursive(
            op: &OpNodeShared,
            visited: &mut HashSet<usize>,
            out: &mut Vec<OpNodeShared>,
        ) {
            fn ptr_as_usize(op: &OpNodeShared) -> usize {
                (op.as_ref() as *const OpNode) as usize
            }
            if !visited.contains(&ptr_as_usize(&op)) {
                visited.insert(ptr_as_usize(&op));
                op._inp.iter().for_each(|prev_value| {
                    if let Some(from_op) = &prev_value.borrow()._parent_op {
                        topo_sort_recursive(from_op, visited, out);
                    };
                });
                out.push(op.clone());
            }
        }

        let mut visited: HashSet<usize> = HashSet::new();
        let mut sorted: Vec<OpNodeShared> = Vec::new();

        let timer = Instant::now();
        if let Some(from_op) = &root.borrow().parent_op() {
            topo_sort_recursive(from_op, &mut visited, &mut sorted);
        }
        println!(
            "Collection of {} nodes took {} ms",
            visited.len(),
            timer.elapsed().as_millis()
        );

        sorted
    }
}
