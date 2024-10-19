use neuronfun::retext::char_set::CharSet;
use neuronfun::nnetwork::FloatType;
use neuronfun::retext::ReText;

fn main() {
    let data = CharSet::new("./datasets/names.txt", 0.9, true);
    let cycles = 100;
    let learning_rate = 0.1 as FloatType;
    let training_batch_size = 1000;
    let block_size = 3;
    let n_hidden_layers = 2;
    let embed_dim = Some(2);
    let layer_size = 30;
    let regularization = None;
    let verbose = true;
    let prediction_seed = "steph";
    let prediction_length = 100;

    let mut retext = ReText::new(
        data,
        block_size,
        embed_dim,
        n_hidden_layers,
        layer_size,
        regularization,
    );
    let text_no_training = retext.predict(prediction_seed, prediction_length).unwrap();

    match retext.import_parameters("names.param"){
        Ok(_) => (),
        Err(err) => match err.kind() {
            std::io::ErrorKind::NotFound | std::io::ErrorKind::UnexpectedEof => {
                eprintln!("Parameter import failed, using randomly initialized parameters instead.")
            }
            _ => panic!("Parameter import failed: {}", err),
        },
    }
    retext.train(cycles, learning_rate, training_batch_size, verbose);
    if let Ok(filename) = retext.export_parameters("out.param") {
        println!("Exported parameters to: {filename}");
    }

    println!("No training: {}", text_no_training);
    println!("With training:");
    for _ in 0..10 {
        let text_with_training = retext.predict(prediction_seed, prediction_length).unwrap();
        println!("{text_with_training}");
    }
}
