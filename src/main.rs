use retext::data_preparing::data_set::DataSet;

use retext::nnetwork::{FloatType, ReText};

fn main() {
    let data = DataSet::new("./datasets/tiny_shakespeare.txt", 1.0, true);
    let cycles = 1000;
    let learning_rate = 1 as FloatType;
    let training_batch_size = 1000;
    let data_block_size = 1;
    let n_hidden_layers = 0;
    let embed_dim = Some(2);
    let layer_size = 100;
    let regularization: Option<FloatType> = None;
    let verbose = true;
    let prediction_seed = "Once upon a time ";
    let prediction_length = 100;

    let mut retext = ReText::new(
        data,
        training_batch_size - data_block_size,
        data_block_size,
        embed_dim,
        n_hidden_layers,
        layer_size,
        regularization
    );
    let text_no_training = retext
        .predict(prediction_seed, prediction_length)
        .unwrap();

    if let Err(err) = retext.import_parameters("shakespeare.param") {
        match err.kind() {
            std::io::ErrorKind::NotFound | std::io::ErrorKind::UnexpectedEof => 
                println!("Parameter import failed, using randomly initialized parameters instead."),
            _ => panic!("Parameter import failed: {}", err),
        }
    }
    retext.train(
        cycles,
        learning_rate,
        training_batch_size,
        verbose,
    );
    retext.export_parameters("shakespeare.param").unwrap();

    let text_with_training = retext
        .predict(prediction_seed, prediction_length)
        .unwrap();

    println!("No training: {}", text_no_training);
    println!("With training: {}", text_with_training);
}
