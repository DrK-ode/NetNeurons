use retext::data_set::DataSet;
use retext::nnetwork::Bigram;

fn main() {
    let data_set = DataSet::new("./datasets/tiny_shakespeare.txt", 1.0);
    let mut bigram_model = Bigram::new(data_set, 1);
    let cycles = 100;
    let learning_rate = 5.;
    let data_block_size = 20;
    let regularization = 0.01;
    let prediction_seed = "Once upon a time ";
    let prediction_length = 100;

    let text_no_training = bigram_model
        .predict(prediction_seed, prediction_length)
        .unwrap();

    bigram_model.learn(cycles, learning_rate, data_block_size, regularization);

    let text_with_training = bigram_model
        .predict(&prediction_seed, prediction_length)
        .unwrap();

    println!("No training: {}", text_no_training);
    println!("With training: {}", text_with_training);
}
