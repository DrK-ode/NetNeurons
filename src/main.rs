use std::str::FromStr;

use retext::data_set::DataSet;
use retext::nnetwork::neurons::NnVec;
use retext::nnetwork::{CharSet, Forward, GradVal, LinearLayer, Neuron, MLP};

fn main() {
    let data_set = DataSet::new("./datasets/tiny_shakespeare.txt", 0.9);
    let characters = CharSet::from_str(data_set.get_training_data()).unwrap();
    println!(
        "Characters present in Shakespeare: {}",
        characters.to_string()
    );

    let mut mlp = MLP::from_empty();
        mlp.add_layer(Box::new(LinearLayer::from_vec(vec![
            Neuron::from_value(1., 2, None),
            Neuron::from_value(1., 2, None),
            Neuron::from_value(1., 2, None),
        ])));
        mlp.add_layer(Box::new(LinearLayer::from_vec(vec![
            Neuron::from_value(1., 3, None),
            Neuron::from_value(1., 3, None),
            Neuron::from_value(1., 3, None),
        ])));
        mlp.add_layer(Box::new(LinearLayer::from_vec(vec![
            Neuron::from_value(1., 3, None),
            Neuron::from_value(1., 3, None),
        ])));
        let input = NnVec::from_vec( vec![GradVal::from(1.0), GradVal::from(2.0)]);
        let output = mlp.forward(&input);
        println!("{input} {mlp} {output}")
}
