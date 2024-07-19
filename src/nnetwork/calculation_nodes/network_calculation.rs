use std::{collections::HashSet, time::Instant};

use super::*;

struct NetworkCalculation {
    _op_order: Vec<OpNodeShared>,
}

impl NetworkCalculation {
    fn new(root: &TensorShared) -> Self {
        NetworkCalculation {
            _op_order: Self::topo_sort(root),
        }
    }

    fn forward(&self) -> TensorShared {
        todo!()
    }
}

impl NetworkCalculation {
    pub fn topo_sort(root: &TensorShared) -> Vec<OpNodeShared> {
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
