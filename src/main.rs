use retext::data_set::DataSet;
use retext::nnetwork::{Bigram, FloatType};

fn main() {
    let data_set = DataSet::new("./datasets/tiny_shakespeare.txt", 1.0, true);
    let mut bigram_model = Bigram::new(data_set, 1, true);
    let cycles = 1000000;
    let learning_rate = 0.05 as FloatType;
    let data_block_size = 128;
    let regularization: Option<FloatType> = None;
    let verbose = true;
    let prediction_seed = "Once upon a time ";
    let prediction_length = 100;

    let text_no_training = bigram_model
        .predict(prediction_seed, prediction_length)
        .unwrap();

    bigram_model.learn(cycles, learning_rate, data_block_size, regularization, verbose);

    let text_with_training = bigram_model
        .predict(prediction_seed, prediction_length)
        .unwrap();

    println!("No training: {}", text_no_training);
    println!("With training: {}", text_with_training);
}
