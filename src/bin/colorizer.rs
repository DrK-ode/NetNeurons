use std::ops::Range;

use neuronfun::{
    colorize::{Color, ColorPredictor},
    nnetwork::FloatType,
};
use plotters::{
    chart::ChartBuilder,
    prelude::{BitMapBackend, IntoDrawingArea},
    style::{BLACK, BLUE, RED, WHITE},
};

fn main() {
    let n_hidden_layers = 2;
    let layer_size = 30;
    let regularization = None;
    
    let mut categorize = ColorPredictor::new(
        Box::new(|(x, y)| (x < 0., y < 0.)),
        n_hidden_layers,
        layer_size,
        regularization,
    );

    let training_cycles = 1000;
    let batch_size = 100;
    let learning_rate = 0.1;
    let x_range = -1. ..1.;
    let y_range = -1. ..1.;
    let verbose = true;
    
    categorize.train(
        training_cycles,
        batch_size,
        learning_rate,
        &x_range,
        &y_range,
        verbose,
    );

    plot_predictions(&categorize, &x_range, &y_range).unwrap();
}

fn plot_predictions(
    predictor: &ColorPredictor,
    x_range: &Range<FloatType>,
    y_range: &Range<FloatType>,
) -> Result<(), Box<dyn std::error::Error>> {
    let drawing_area = BitMapBackend::new("plot.png", (800, 600)).into_drawing_area();
    drawing_area.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&drawing_area)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(x_range.start..x_range.end, y_range.start..y_range.end)?;
    chart.configure_mesh().light_line_style(WHITE).draw()?;

    let pixel_range = chart.plotting_area().get_pixel_range();
    let size = (
        pixel_range.0.end - pixel_range.0.start,
        pixel_range.1.end - pixel_range.1.start,
    );
    let range = (
        chart.plotting_area().get_x_range(),
        chart.plotting_area().get_y_range(),
    );
    let step = (
        (range.0.end - range.0.start) / (size.0 + 1) as FloatType,
        (range.1.end - range.1.start) / (size.1 + 1) as FloatType,
    );
    for x_pixel in pixel_range.0 {
        let x = range.0.start + step.0 * x_pixel as FloatType;
        for y_pixel in pixel_range.1.clone() {
            let y = range.1.start + step.1 * y_pixel as FloatType;
            println!("({x},{y})");
            let color = match predictor.predict((x, y)) {
                Color::Red => &RED,
                Color::Blue => &BLUE,
                Color::None => &BLACK,
                Color::Both => &WHITE,
            };
            chart.plotting_area().draw_pixel((x, y), color)?;
        }
    }

    drawing_area.present()?;
    Ok(())
}
