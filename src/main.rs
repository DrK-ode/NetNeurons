use std::str::FromStr;

use retext::data_set::DataSet;
use retext::nnetwork::{CharSet, Forward, GradVal, MLP, NnVec};

fn main() {
    let data_set = DataSet::new("./datasets/tiny_shakespeare.txt", 0.9);
    let characters = CharSet::from_str(data_set.get_training_data()).unwrap();
    println!(
        "Characters present in Shakespeare: {}",
        characters.to_string()
    );

    let mlp = MLP::new(3, 4, 2, 3);
    let input = NnVec::from_vec(vec![GradVal::from(1.0), GradVal::from(2.0)]);
    let output = mlp.forward(&input);
    println!("{input} {mlp} {output}")
}
