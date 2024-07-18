use super::{op_node::OpNodeShared, val_node::ValNodeShared};

struct NetworkCalculation{
    _op_order: Vec<OpNodeShared>,
}

impl NetworkCalculation{
    fn new(root: &ValNodeShared) -> Self {
        NetworkCalculation{
        _op_order: topo_sort(root),
        }
    }

    fn forward(&self) -> ValNodeShared {
        todo!()
    }
}

fn topo_sort(_root: &ValNodeShared) -> Vec<OpNodeShared> {
    todo!()
}