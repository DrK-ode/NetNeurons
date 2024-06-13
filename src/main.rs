use std::str::FromStr;

use retext::data_set::DataSet;

use retext::nnetwork::CharSet;

fn main() {
    let data_set = DataSet::new("./datasets/tiny_shakespeare.txt",0.9);
    let characters = CharSet::from_str(data_set.get_training_data()).unwrap();
    println!("Characters present in Shakespeare: {}", characters.to_string());
}
