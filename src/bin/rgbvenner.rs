use std::ops::Range;

use neuronfun::{color_selector::ColorSelector, nnetwork::FloatType};
use plotters::{
    chart::{ChartBuilder, LabelAreaPosition},
    prelude::{BitMapBackend, Cross, IntoDrawingArea, Rectangle},
    style::{RGBAColor, ShapeStyle, RED, WHITE},
};

fn main() {
    let n_hidden_layers = 3;
    let layer_size = 20;
    let regularization = None;

    let mut categorize = ColorSelector::new(
        Box::new(|(x, y)| {
            [
                (x - 0.2165).powi(2) + (y+0.125).powi(2) < 0.25,
                (x + 0.2165).powi(2) + (y+0.125).powi(2) < 0.25,
                x.powi(2) + (y - 0.25).powi(2) < 0.25,
            ]
        }),
        n_hidden_layers,
        layer_size,
        regularization,
    );

    match categorize.import_parameters("rgb.param") {
        Ok(_) => println!("Successful parameter import from rgb.param."),
        Err(err) => match err.kind() {
            std::io::ErrorKind::NotFound | std::io::ErrorKind::UnexpectedEof => {
                eprintln!("Parameter import failed, using randomly initialized parameters instead.")
            }
            _ => panic!("Parameter import failed: {}", err),
        },
    }

    let training_cycles = 100000;
    let batch_size = 100;
    let learning_rate = 0.1..0.01; // Logspaced
    let x_range = -1. ..1.;
    let y_range = -1. ..1.;
    let verbose = true;

    let training_data = categorize.train(
        training_cycles,
        batch_size,
        learning_rate,
        &x_range,
        &y_range,
        verbose,
    );

    if let Ok(filename) = categorize.export_parameters("rgb.param") {
        println!("Exported RGB parameters to {filename}");
    }

    plot_predictions(&categorize, &x_range, &y_range, 500).unwrap();
    plot_training_progress(&training_data).unwrap();
}

fn plot_predictions(
    predictor: &ColorSelector,
    x_range: &Range<FloatType>,
    y_range: &Range<FloatType>,
    x_divisions: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let size = (x_range.end - x_range.start, y_range.end - y_range.start);
    let division = (x_divisions, (x_divisions as f64 * size.1 / size.0) as usize);
    let step = (
        size.0 / division.0 as FloatType,
        size.1 / division.1 as FloatType,
    );

    const X_PIXELS: u32 = 1000;
    let drawing_area = BitMapBackend::new(
        "plot_rgb.png",
        (X_PIXELS, (X_PIXELS as f64 * size.1 / size.0) as u32),
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
        .fold((f64::MAX, f64::MAX), |acc, (x, y)| {
            (x.min(acc.0), y.min(acc.1))
        });
    let max = training_data
        .iter()
        .fold((f64::MIN, f64::MIN), |acc, (x, y)| {
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
