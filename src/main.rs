use retext::data_preparing::data_set::DataSet;

use retext::nnetwork::{Bigram, FloatType};

fn main() {
    let data_set = DataSet::new("./datasets/tiny_shakespeare.txt", 1.0, true);
    let cycles = 1000;
    let learning_rate = 0.1 as FloatType;
    let training_batch_size = 128;
    let _data_block_size = 3;
    let number_of_layers = 1;
    let regularization: Option<FloatType> = None;
    let verbose = true;
    let prediction_seed = "Once upon a time ";
    let prediction_length = 100;

    let mut bigram_model = Bigram::new(data_set, number_of_layers, true);
    let text_no_training = bigram_model
        .predict(prediction_seed, prediction_length)
        .unwrap();

    bigram_model.import_parameters("shakespeare.param").unwrap();
    bigram_model.learn(cycles, learning_rate, training_batch_size, regularization, verbose);
    bigram_model.export_parameters("shakespeare.param").unwrap();

    let text_with_training = bigram_model
        .predict(prediction_seed, prediction_length)
        .unwrap();

    println!("No training: {}", text_no_training);
    println!("With training: {}", text_with_training);
}
