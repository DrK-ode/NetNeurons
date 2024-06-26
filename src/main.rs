use retext::data_set::DataSet;
use retext::nnetwork::Bigram;

fn main() {
    let data_set = DataSet::new("./datasets/tiny_shakespeare.txt", 0.9);
    let bigram_model = Bigram::new( data_set, 1 );
}
