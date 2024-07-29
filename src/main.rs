use retext::data_preparing::data_set::DataSet;

use retext::nnetwork::{FloatType, ParameterBundle, ReText};

fn main() {
    let data = DataSet::new("./datasets/tiny_shakespeare.txt", 1.0, true);
    let cycles = 1000;
    let learning_rate = 0.001 as FloatType;
    let training_batch_size = 1024;
    let block_size = 3;
    let n_hidden_layers = 3;
    let embed_dim = Some(10);
    let layer_size = 200;
    let regularization: Option<FloatType> = None;
    let verbose = true;
    let prediction_seed = "Once upon a time ";
    let prediction_length = 100;

    let mut retext = ReText::new(
        data,
        training_batch_size - block_size,
        block_size,
        embed_dim,
        n_hidden_layers,
        layer_size,
        regularization
    );
    let text_no_training = retext
        .predict(prediction_seed, prediction_length)
        .unwrap();

    match ParameterBundle::import_parameters("shakespeare.param") {
        Ok(bundle) => {
            retext.load_trainer_parameter_bundle(&bundle);
        }
        Err(err) =>
        match err.kind() {
            std::io::ErrorKind::NotFound | std::io::ErrorKind::UnexpectedEof => 
                eprintln!("Parameter import failed, using randomly initialized parameters instead."),
            _ => panic!("Parameter import failed: {}", err),
        }
    }
    retext.train(
        cycles,
        learning_rate,
        training_batch_size,
        verbose,
    );
    let bundle = retext.get_parameter_bundle();
    bundle.export_parameters("shakespeare.param").unwrap();
    retext.load_predictor_parameter_bundle(&bundle);

    let text_with_training = retext
        .predict(prediction_seed, prediction_length)
        .unwrap();

    println!("No training: {}", text_no_training);
    println!("With training: {}", text_with_training);
}
