use std::ops::Range;

use net_neurons::{nnetwork::FloatType, recolor::{color_key::{ColorKey, RGB_VENN_DIAGRAM}, ColorSelector}};
use plotters::{
    chart::{ChartBuilder, LabelAreaPosition},
    prelude::{BitMapBackend, Cross, IntoDrawingArea, Rectangle},
    style::{RGBAColor, ShapeStyle, RED, WHITE},
};

fn main() {
    // Specifies wheter a pixel is red, green and/or blue
    const COLOR_KEY: ColorKey = RGB_VENN_DIAGRAM;
    // Every neural network will have an input and output layer. The hidden layers are inbetween these.
    const N_HIDDEN_LAYERS: usize = 3;
    // The layers are square so the number of neurons is this number squared
    const LAYER_SIZE: usize = 20;
    // Set to some to punish non-zero parameters.
    const REGULARIZATION: Option<FloatType> = None;

    // Instantiate the network
    let mut categorize = ColorSelector::new(COLOR_KEY, N_HIDDEN_LAYERS, LAYER_SIZE, REGULARIZATION);

    // Import previously exported parameters if able. Will fallback to random initiated neurons if the file does not exist, but panic on other errors.
    const IMPORT_FILENAME: &str = "rgb.param";
    match categorize.import_parameters(IMPORT_FILENAME) {
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
    const TRAINING_CYCLES: usize = 100000;
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
    let training_data = categorize.train(
        TRAINING_CYCLES,
        TRAINING_BATCH_SIZE,
        LEARNING_RATE,
        &X_RANGE,
        &Y_RANGE,
        VERBOSE,
    );

    // Save the resulting network. Will overwrite any existing file!
    const EXPORT_FILENAME: &str = "rgb.param";
    if let Ok(filename) = categorize.export_parameters(EXPORT_FILENAME) {
        println!("Exported RGB parameters to {filename}");
    }

    // Plots the colours predicted by the network for a sample of coordinates
    const X_SAMPLES: u32 = 500;
    plot_predictions(&categorize, &X_RANGE, &Y_RANGE, X_SAMPLES).unwrap();
    // Plots a diagram of log(loss) vs p(learning rate)
    plot_training_progress(&training_data).unwrap();
}

fn plot_predictions(
    predictor: &ColorSelector,
    x_range: &Range<FloatType>,
    y_range: &Range<FloatType>,
    x_divisions: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let size = (x_range.end - x_range.start, y_range.end - y_range.start);
    let division = (
        x_divisions,
        (x_divisions as FloatType * size.1 / size.0) as usize,
    );
    let step = (
        size.0 / division.0 as FloatType,
        size.1 / division.1 as FloatType,
    );

    const X_PIXELS: u32 = 1000;
    let drawing_area = BitMapBackend::new(
        "plot_rgb.png",
        (X_PIXELS, (X_PIXELS as FloatType * size.1 / size.0) as u32),
    )
    .into_drawing_area();
    drawing_area.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&drawing_area)
        .x_label_area_size(0)
        .y_label_area_size(0)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .build_cartesian_2d(x_range.start..x_range.end, y_range.start..y_range.end)?;

    chart.draw_series((0..division.0).flat_map(|xi| {
        let xl = x_range.start + step.0 * xi as FloatType;
        let xr = xl + step.0;
        let xm = xl + step.0 * 0.5;
        (0..division.1).map(move |yi| {
            let yt = y_range.start + step.1 * yi as FloatType;
            let yb = yt + step.1;
            let ym = yt + step.1 * 0.5;
            let rgb: Vec<_> = predictor
                .predict((xm, ym))
                .iter()
                .map(|c| (c * 255.0) as u8)
                .collect();
            let color = RGBAColor(rgb[0], rgb[1], rgb[2], 1.);
            Rectangle::new(
                [(xl, yt), (xr, yb)],
                ShapeStyle {
                    color,
                    filled: true,
                    stroke_width: 0,
                },
            )
        })
    }))?;
    chart.configure_mesh().disable_mesh().draw()?;

    drawing_area.present()?;
    Ok(())
}

fn plot_training_progress(
    training_data: &[(FloatType, FloatType)],
) -> Result<(), Box<dyn std::error::Error>> {
    const X_PIXELS: u32 = 1024;
    const Y_PIXELS: u32 = 768;
    let drawing_area =
        BitMapBackend::new("plot_loss.png", (X_PIXELS, Y_PIXELS)).into_drawing_area();
    let min = training_data
        .iter()
        .fold((FloatType::MAX, FloatType::MAX), |acc, (x, y)| {
            (x.min(acc.0), y.min(acc.1))
        });
    let max = training_data
        .iter()
        .fold((FloatType::MIN, FloatType::MIN), |acc, (x, y)| {
            (x.max(acc.0), y.max(acc.1))
        });
    let x_begin = -max.0.log10();
    let x_end = -min.0.log10();
    let y_begin = min.1.log10();
    let y_end = max.1.log10();
    drawing_area.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&drawing_area)
        .x_label_area_size(0)
        .y_label_area_size(0)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .build_cartesian_2d(x_begin..x_end, y_begin..y_end)?;

    chart.draw_series(
        training_data.iter().map(|(learning_rate, loss)| {
            Cross::new((-learning_rate.log10(), loss.log10()), 5, RED)
        }),
    )?;
    chart
        .configure_mesh()
        .x_desc("Neg log10 learning rate")
        .y_desc("Log10 Loss")
        .draw()?;

    drawing_area.present()?;
    Ok(())
}
