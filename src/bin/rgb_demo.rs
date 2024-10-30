use std::ops::Range;

use net_neurons::{
    nnetwork::FloatType,
    recolor::ReColor,
};

fn main() {
    // Example "color key" representing three overlapping spheres.
    let rgb_venn_diagram = &|(x, y): (FloatType, FloatType)| {
        [
            (x - 0.2165).powi(2) + (y + 0.125).powi(2) < 0.25,
            (x + 0.2165).powi(2) + (y + 0.125).powi(2) < 0.25,
            x.powi(2) + (y - 0.25).powi(2) < 0.25,
        ]
    };
    // Every neural network will have an input and output layer. The hidden layers are inbetween these.
    const N_HIDDEN_LAYERS: usize = 3;
    // The layers are square so the number of neurons is this number squared
    const LAYER_SIZE: usize = 20;
    // Set to some to punish non-zero parameters.
    const REGULARIZATION: Option<FloatType> = None;

    // Instantiate the network
    let mut recolor = ReColor::new(
        rgb_venn_diagram,
        N_HIDDEN_LAYERS,
        LAYER_SIZE,
        REGULARIZATION,
    );

    // Import previously exported parameters if able. Will fallback to random initiated neurons if the file does not exist, but panic on other errors.
    const IMPORT_FILENAME: &str = "rgb.param";
    match recolor.import_parameters(IMPORT_FILENAME) {
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
    const TRAINING_CYCLES: usize = 50000;
    // The number of data points to use before updating the network by back propagation
    const TRAINING_BATCH_SIZE: usize = 100;
    // The range of learning rates to be used. Will be logspaced.
    const LEARNING_RATE: Range<FloatType> = 0.1..0.01;
    // Limit the input coordinate space
    const X_RANGE: Range<FloatType> = -1. ..1.;
    const Y_RANGE: Range<FloatType> = -1. ..1.;
    // Lots of text...or not
    const VERBOSE: bool = true;
    // Returns a vector of learning rates and loss values
    recolor.train(
        TRAINING_CYCLES,
        TRAINING_BATCH_SIZE,
        LEARNING_RATE,
        &X_RANGE,
        &Y_RANGE,
        VERBOSE,
    );

    // Save the resulting network. Will overwrite any existing file!
    const EXPORT_FILENAME: &str = "rgb.param";
    if let Ok(filename) = recolor.export_parameters(EXPORT_FILENAME) {
        println!("Exported RGB parameters to {filename}");
    }

    // Plots the colours predicted by the network for a sample of coordinates
    const X_SAMPLES: u32 = 500;
    const PLOT_PREDICTION_FILENAME: &str = "plot_rgb.png";
    recolor.plot_predictions(&X_RANGE, &Y_RANGE, X_SAMPLES, PLOT_PREDICTION_FILENAME).unwrap();
    // Plots a diagram of log(loss) vs p(learning rate)
    const PLOT_PROGRESS_FILENAME: &str = "plot_loss.png";
    recolor.plot_training_progress(PLOT_PROGRESS_FILENAME).unwrap();
}
