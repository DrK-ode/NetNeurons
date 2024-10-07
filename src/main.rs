use plotters::chart::ChartBuilder;
use plotters::prelude::{BitMapBackend, Circle, EmptyElement, IntoDrawingArea, Text};
use plotters::style::{IntoFont, BLACK, BLUE, RED, WHITE};
use retext::data_preparing::data_set::DataSet;

use retext::nnetwork::{FloatType, ParameterBundle, ReText};

fn main() {
    let mut data = DataSet::new("./datasets/names.txt", 0.9, true);
    data.add_character('^');
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
        //training_batch_size,
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
    if let Ok(filename) = bundle.export_parameters("out.param") {
        println!("Exported parameters to: {filename}");
    }

    println!("No training: {}", text_no_training);
    println!("With training:");
    for _ in 0..10 {
        let text_with_training = retext.predict(prediction_seed, prediction_length).unwrap();
        println!("{text_with_training}");
    }

    if let Some(dim) = embed_dim {
        if dim == 2 {
            plot_embedding(&retext).unwrap();
        }
    }
}

// Only really makes sense for 2D embedding.
fn plot_embedding(retext: &ReText) -> Result<(), Box<dyn std::error::Error>> {
    let drawing_area = BitMapBackend::new("plot.png", (2048, 2048)).into_drawing_area();
    drawing_area.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&drawing_area)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(-3.0..3.0, -3.0..3.0)?;
    chart.configure_mesh().light_line_style(WHITE).draw()?;
    let font = ("sans-serif", 20).into_font().resize(20.).color(&BLACK);
    chart.draw_series(retext.characters().iter().map(|letter| {
        let color = match &letter {
            'a' | 'e' | 'i' | 'o' | 'u' => RED,
            '^' | ' ' => BLACK,
            _ => BLUE,
        };
        let letter = letter.to_string();
        let coords = retext.embed(&letter).copy_vals();
        //println!("{letter} -> {:?}", coords);
        let empty = EmptyElement::at((coords[0], coords[1]));
        let circle = Circle::new((0, 0), 2, color);
        let text = Text::new(letter, (5, 5), &font);
        empty + circle + text
    }))?;
    drawing_area.present()?;
    Ok(())
}
