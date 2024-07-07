use retext::data_set::DataSet;
use retext::nnetwork::Bigram;

fn main() {
    let data_set = DataSet::new("./datasets/tiny_shakespeare.txt", 0.9);
    let mut bigram_model = Bigram::new( data_set, 1 );
    let cycles = 100;
    let learning_rate = 0.001;
    let data_block_size = 100;
    bigram_model.learn(cycles, learning_rate, data_block_size);

    let number_of_characters = 10;
    let text = bigram_model.predict("Once upon a time ", number_of_characters);

    println!("{}", text.unwrap());
}
