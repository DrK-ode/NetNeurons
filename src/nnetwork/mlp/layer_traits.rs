use std::fmt::Display;
use std::io::{Read, Write};
use std::{fs::File, iter::empty};

use crate::nnetwork::calculation_nodes::TensorShared;
use crate::nnetwork::{FloatType, TensorShape};

pub trait Forward {
    fn forward(&self, inp: &TensorShared) -> TensorShared;
}

pub trait Parameters {
    fn parameters(&self) -> Box<dyn Iterator<Item = &TensorShared> + '_> {
        Box::new(empty())
    }

    // Adds a numerical suffix if the wanted filename is taken. The filename is returned upon successful export.
    fn export_parameters(&self, filename: &str) -> std::io::Result<String> {
        let mut fn_string = filename.to_string();
        let mut counter: usize = 0;
        let mut file = loop {
            let file = File::create_new(&fn_string);
            match file {
                Ok(file) => {
                    if counter > 0 {
                        println!("Exporting parameters to; {fn_string}");
                    }
                    break file;
                }
                Err(err) => match err.kind() {
                    std::io::ErrorKind::AlreadyExists => (),
                    _ => {
                        println!("Export parameters failed: {}", err)
                    }
                },
            }
            fn_string = filename.to_string() + "." + &counter.to_string();
            counter += 1;
        };
        self.parameters().for_each(|param| {
            param
                .borrow()
                .value()
                .iter()
                .for_each(|v| file.write_all(v.to_le_bytes().as_slice()).unwrap());
        });
        Ok(fn_string)
    }

    fn import_parameters(&self, filename: &str) -> std::io::Result<()> {
        match File::open(filename) {
            Ok(mut file) => {
                let buffer = &mut [0u8; std::mem::size_of::<FloatType>()];
                for param in self.parameters() {
                    let mut vec = vec![f64::NAN; param.len()];
                    for v in vec.iter_mut() {
                        match file.read_exact(buffer) {
                            Ok(_) => *v = FloatType::from_le_bytes(*buffer),
                            Err(err) => {
                                return Err(err);
                            }
                        }
                    }
                    param.borrow_mut().set_value(vec);
                }
                Ok(())
            }
            Err(err) => {
                println!("Import parameters failed: {}", err);
                Err(err)
            }
        }
    }
}

pub trait Layer: Forward + Parameters + Display {
    fn shape(&self) -> Option<TensorShape> {
        None
    }
}
