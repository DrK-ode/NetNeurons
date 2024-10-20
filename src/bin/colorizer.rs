use std::{f32::consts::PI, ops::Range};

use neuronfun::{
    colorize::{Color, ColorPredictor},
    nnetwork::FloatType,
};
use plotters::{
    chart::{ChartBuilder, LabelAreaPosition},
    prelude::{BitMapBackend, IntoDrawingArea, Rectangle},
    style::{ShapeStyle, BLACK, BLUE, MAGENTA, RED, WHITE},
};

fn main() {
    let n_hidden_layers = 3;
    let layer_size = 20;
    let regularization = None;

    let mut categorize = ColorPredictor::new(
        Box::new(|(x, y)| (x.sin() < y, x.cos() > y)),
        n_hidden_layers,
        layer_size,
        regularization,
    );

    let training_cycles = 1000;
    let batch_size = 100;
    let learning_rate = 1. ..0.001;
    let x_range = -PI as FloatType..PI as FloatType;
    let y_range = -1.5..1.5;
    let verbose = true;

    categorize.train(
        training_cycles,
        batch_size,
        learning_rate,
        &x_range,
        &y_range,
        verbose,
    );

    plot_predictions(&categorize, &x_range, &y_range, 500).unwrap();
}

fn plot_predictions(
    predictor: &ColorPredictor,
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
        "plot.png",
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
            let color = match predictor.predict((xm, ym)) {
                Color::Red => RED,
                Color::Blue => BLUE,
                Color::None => WHITE,
                Color::Both => MAGENTA,
            };
            Rectangle::new(
                [(xl, yt), (xr, yb)],
                ShapeStyle {
                    color: color.into(),
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
