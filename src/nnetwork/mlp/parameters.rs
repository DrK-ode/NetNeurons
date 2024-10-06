use std::fs::read_to_string;
use std::io::Write;
use std::{fs::File, iter::empty};

use crate::nnetwork::{CalcNodeShared, FloatType};

use super::Layer;

pub trait Parameters {
    fn param_iter(&self) -> Box<dyn Iterator<Item = &CalcNodeShared> + '_> {
        Box::new(empty())
    }
}

#[derive(Debug, PartialEq)]
pub struct ParameterBundle {
    _parameters: Vec<(String, Vec<Vec<FloatType>>)>,
}

impl ParameterBundle {
    pub fn new(layers: &Vec<Box<dyn Layer>>) -> ParameterBundle {
        let mut parameters = Vec::new();
        for layer in layers {
            let mut layer_vec = Vec::new();
            for param in layer.param_iter() {
                layer_vec.push(param.copy_vals());
            }
            parameters.push((layer.layer_name().to_owned(), layer_vec));
        }
        ParameterBundle {
            _parameters: parameters,
        }
    }

    pub fn load_parameters(&self, layers: &Vec<Box<dyn Layer>>) {
        for ((layer_name_stored, layer_stored), layer) in self._parameters.iter().zip(layers) {
            if layer_name_stored != layer.layer_name() {
                eprintln!(
                    "Warning, layer name {} do not match stored layer name {}",
                    layer.layer_name(),
                    layer_name_stored
                );
            }
            let n_param = layer.param_iter().count();
            assert_eq!(layer_stored.len(), n_param,"Error, number of parameters {} in layer {} do not match stored number of parameters {}", n_param, layer.layer_name(), layer_stored.len());
            for (param_stored, param) in layer_stored.iter().zip(layer.param_iter()) {
                assert_eq!(
                    param_stored.len(),
                    param.len(),
                    "Error, size {} of parameter do not match stored size {}",
                    param.len(),
                    param_stored.len()
                );
                param.set_vals(param_stored);
            }
        }
    }

    // Adds a numerical suffix if the wanted filename is taken. The filename is returned upon successful export.
    pub fn export_parameters(&self, filename: &str) -> std::io::Result<String> {
        let mut fn_string = filename.to_string();
        let mut counter: usize = 0;
        let mut file = loop {
            let file = File::create_new(&fn_string);
            match file {
                Ok(file) => {
                    if counter > 0 {
                        eprintln!("Changing export filename to; {fn_string}");
                    }
                    break file;
                }
                Err(err) => match err.kind() {
                    std::io::ErrorKind::AlreadyExists => (),
                    _ => {
                        eprintln!("Export parameters failed: {}", err)
                    }
                },
            }
            fn_string = filename.to_string() + "." + &counter.to_string();
            counter += 1;
        };
        for (i, (name, layer_params)) in self._parameters.iter().enumerate() {
            writeln!(file, "Layer {i}: {name}")?;
            for (n, param) in layer_params.iter().enumerate() {
                writeln!(file, "Parameter: {n}")?;
                for value in param {
                    writeln!(file, "{value}")?;
                }
            }
        }
        Ok(fn_string)
    }

    pub fn import_parameters(filename: &str) -> std::io::Result<ParameterBundle> {
        let mut parameters = Vec::new();
        let mut current_layer_vec = Vec::new();
        let mut current_param_vec = Vec::new();
        let mut current_name = String::new();
        let file_content = read_to_string(filename);
        match file_content {
            Ok(content) => {
                for line in content.lines() {
                    if line.starts_with("Layer") {
                        if !current_param_vec.is_empty() {
                            current_layer_vec.push(current_param_vec);
                            current_param_vec = Vec::new();
                        }
                        if !current_name.is_empty() {
                            parameters.push((current_name, current_layer_vec));
                            current_layer_vec = Vec::new()
                        }
                        current_name = line.split_once(": ").unwrap().1.to_string();
                    } else if line.starts_with("Parameter") {
                        if !current_param_vec.is_empty() {
                            current_layer_vec.push(current_param_vec);
                            current_param_vec = Vec::new();
                        }
                    } else {
                        current_param_vec.push(line.parse().unwrap())
                    }
                }
                if !current_layer_vec.is_empty() {
                    parameters.push((current_name, current_layer_vec));
                }
                Ok(ParameterBundle {
                    _parameters: parameters,
                })
            }
            Err(err) => Err(err),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::nnetwork::{FunctionLayer, LinearLayer};

    use super::*;

    #[test]
    fn parameter_bundle_transfer() {
        let layers: Vec<Box<dyn Layer>> = vec![
            Box::new(LinearLayer::from_nodes(
                CalcNodeShared::filled_from_shape((3,2), vec![1., 2., 3., 4., 5., 6.]),
                None,
                "TestLayer",
            )),
            Box::new(FunctionLayer::new(
                &FunctionLayer::tanh,
                "tanh",
                "TestLayer",
            )),
            Box::new(LinearLayer::from_nodes(
                CalcNodeShared::filled_from_shape((2,3),vec![1., -1., 1., -1., 1., -1.]),
                None,
                "TestLayer",
            )),
        ];
        let layers_copy: Vec<Box<dyn Layer>> = vec![
            Box::new(LinearLayer::from_nodes(
                CalcNodeShared::filled_from_shape((3,2),vec![0.; 6]),
                None,
                "TestLayer",
            )),
            Box::new(FunctionLayer::new(
                &FunctionLayer::tanh,
                "tanh",
                "TestLayer",
            )),
            Box::new(LinearLayer::from_nodes(
                CalcNodeShared::filled_from_shape((2,3),vec![0.; 6]),
                None,
                "TestLayer",
            )),
        ];
        let bundle = ParameterBundle::new(&layers);
        bundle.load_parameters(&layers_copy);
        let bundle_copy = ParameterBundle::new(&layers_copy);
        assert_eq!(bundle, bundle_copy);
    }
}
