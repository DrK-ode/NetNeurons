use retext::data_set::DataSet;
use retext::nnetwork::Bigram;

fn main() {
    let data_set = DataSet::new("./datasets/test.txt", 1.0);
    let mut bigram_model = Bigram::new(data_set, 2);
    let cycles = 10000;
    let learning_rate = 1.;
    let data_block_size = 5;
    let prediction_seed = "Once upon a time ";
    let prediction_length = 10;

    let text_no_training = bigram_model
        .predict(prediction_seed, prediction_length)
        .unwrap();

    bigram_model.learn(cycles, learning_rate, data_block_size);

    let text_with_training = bigram_model
        .predict(&prediction_seed, prediction_length)
        .unwrap();

    println!("No training: {}", text_no_training);
    println!("With training: {}", text_with_training);
}
