use std::str::FromStr;

use retext::data_set::DataSet;

use retext::nnetwork::{CharSet, GradVal};

fn main() {
    let data_set = DataSet::new("./datasets/tiny_shakespeare.txt",0.9);
    let characters = CharSet::from_str(data_set.get_training_data()).unwrap();
    println!("Characters present in Shakespeare: {}", characters.to_string());

    let x: GradVal = 1_f32.into();
    let y = x.exp();
    let mut z = y.pow(&2_f32.into());
    z.backward();
    println!("{}, {}, {}",x,y,z);
}
