use net_neurons::nnetwork::FloatType;
use net_neurons::retext::CharSet;
use net_neurons::retext::ReText;

fn main() {
    const DATA_FILE: &str = "./datasets/names.txt";
    // Data is read line by line
    let data = CharSet::new(DATA_FILE, 0.9, true);
    // How many previous letters to use in order to predict the next one
    const BLOCK_SIZE: usize = 3;
    // Option<usize>, set to None to disable initial embedding
    const EMBED_DIM: Option<usize> = Some(2);
    // Every neural network will have an input and output layer. The hidden layers are inbetween these.
    const N_HIDDEN_LAYERS: usize = 2;
    // The layers are square so the number of neurons is this number squared
    const HIDDEN_LAYER_SIZE: usize = 30;
    // Set to some to punish non-zero parameters.
    const REGULARIZATION: Option<FloatType> = None;

    // Instantiate the network
    let mut retext = ReText::new(
        data,
        BLOCK_SIZE,
        EMBED_DIM,
        N_HIDDEN_LAYERS,
        HIDDEN_LAYER_SIZE,
        REGULARIZATION,
    );

    // Make a random prediction while the network is untrained
    const PREDICTION_SEED: &str = "steph";
    const PREDICTION_LENGTH: usize = 100;
    let text_no_training = retext.predict(PREDICTION_SEED, PREDICTION_LENGTH).unwrap();

    // Import previously exported parameters if able. Will fallback to random initiated neurons if the file does not exist, but panic on other errors.
    const IMPORT_FILENAME: &str = "names.param";
    match retext.import_parameters(IMPORT_FILENAME) {
        Ok(_) => println!("Successful parameter import from {IMPORT_FILENAME}."),
        Err(err) => match err.kind() {
            std::io::ErrorKind::NotFound | std::io::ErrorKind::UnexpectedEof => {
                eprintln!("Parameter import from {IMPORT_FILENAME} failed, using randomly initialized parameters instead.")
            }
            _ => panic!(
                "Parameter import from {IMPORT_FILENAME} failed catastrophically: {}",
                err
            ),
        },
    }

    // The number of updates to the netowrk
    const TRAINING_CYCLES: usize = 100;
    // Step size will be this fraction times the calculated gradient
    const LEARNING_RATE: FloatType = 0.1;
    // The number of data points to use before updating the network by back propagation
    const TRAINING_BATCH_SIZE: usize = 1000;
    // Lots of text...or not
    const VERBOSE: bool = true;
    retext.train(TRAINING_CYCLES, LEARNING_RATE, TRAINING_BATCH_SIZE, VERBOSE);

    // Save the resulting network. Will overwrite any existing file!
    const EXPORT_FILENAME: &str = "names.param";
    if let Ok(filename) = retext.export_parameters(EXPORT_FILENAME) {
        println!("Exported parameters to: {filename}");
    }

    // Compare untrained to trained network
    println!("No training: {}", text_no_training);
    println!("With training:");
    for _ in 0..10 {
        let text_with_training = retext.predict(PREDICTION_SEED, PREDICTION_LENGTH).unwrap();
        println!("   {text_with_training}");
    }
}
