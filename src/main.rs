use retext::data_preparing::data_set::DataSet;

use retext::nnetwork::{FloatType, ParameterBundle, ReText};

fn main() {
    let mut data = DataSet::new("./datasets/names.txt", 0.8, true);
    data.add_character('^');
    let cycles = 100;
    let learning_rate = 0.01 as FloatType;
    let training_batch_size = 100000;
    let block_size = 1;
    let n_hidden_layers = 0;
    let embed_dim = None;
    let layer_size = 100;
    let regularization = None;
    let verbose = true;
    let prediction_seed = "";
    let prediction_length = 100;

    let mut retext = ReText::new(
        data,
        training_batch_size,
        block_size,
        embed_dim,
        n_hidden_layers,
        layer_size,
        regularization,
    );
    let text_no_training = retext.predict(prediction_seed, prediction_length).unwrap();

    match ParameterBundle::import_parameters("names.param") {
        Ok(bundle) => {
            retext.load_trainer_parameter_bundle(&bundle);
        }
        Err(err) => match err.kind() {
            std::io::ErrorKind::NotFound | std::io::ErrorKind::UnexpectedEof => {
                eprintln!("Parameter import failed, using randomly initialized parameters instead.")
            }
            _ => panic!("Parameter import failed: {}", err),
        },
    }
    retext.train(cycles, learning_rate, training_batch_size, verbose);
    let bundle = retext.get_parameter_bundle();
    bundle.export_parameters("out.param").unwrap();
    retext.load_predictor_parameter_bundle(&bundle);

    println!("No training: {}", text_no_training);
    println!("With training:");
    for _ in 0..10 {
        let text_with_training = retext.predict(prediction_seed, prediction_length).unwrap();
        println!("{text_with_training}");
    }
}
