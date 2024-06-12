use retext::data_import::{import_names, import_shakespeare};

fn main() {
    let (data_training, data_validate) = import_names(0.9);
    println!("Printing 10 out of {} names:", data_training.len()+data_validate.len());
    for name in &data_training[0..10]{
        println!("{}", name);
    }
    println!("");
    println!("Printing Shakespeare:");
    let (data_training, _data_validate) = import_shakespeare(0.9);
        println!("{}", &data_training[0..500]);
    for c in _data_validate.chars(){
        println!("{}",c);
    }
}
