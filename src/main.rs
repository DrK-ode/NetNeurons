use std::str::FromStr;

use retext::data_import::{import_names, import_shakespeare};

use retext::bigram::CharSet;

fn main() {
    let (data_training, _data_validate) = import_names(0.9);
    let characters = CharSet::from_str_vec(&data_training);
    println!("Characters present in names: {}", characters);
    
    let (data_training, _data_validate) = import_shakespeare(0.9);
    let characters = CharSet::from_str(&data_training).unwrap();
    println!("Characters present in Shakespeare: {}", characters);
}
