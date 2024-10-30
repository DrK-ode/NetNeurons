use std::{
    fmt::Display,
    fs::{read_to_string, File},
    io::{Error, Write},
};

use crate::nnetwork::{calc_node::FloatType, CalcNode, Layer, Parameters};

use super::loss_functions::{neg_log_likelihood, LossFuncType};


/// This struct is just a stack of [Layer]s with some convenience attached.
pub struct MultiLayer {
    _layers: Vec<Box<dyn Layer>>,
    _regularization: Option<FloatType>,
    _loss_func: Box<LossFuncType>,
}

impl MultiLayer {
    /// All [Layer]s are constructed beforehand and then put into the [MultiLayer].
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Self {
        MultiLayer {
            _layers: layers,
            _regularization: None,
            _loss_func: Box::new(&neg_log_likelihood),
        }
    }

    /// Use your own or one of the ones provided in the module [crate::nnetwork::mlp::loss_functions].
    pub fn set_loss_function(&mut self, f: &'static LossFuncType) {
        self._loss_func = Box::new(f);
    }
    
    /// Set to `Some(float)` to punish non-zero parameters.
    pub fn set_regularization(&mut self, reg: Option<FloatType>) {
        self._regularization = reg;
    }
    
    /// Returns the number of [Layer]s
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self._layers.len()
    }

    /// Returns the indexed layer. Panics if the index invalid.
    pub fn get_layer(&self, i: usize) -> &dyn Layer {
        assert!(i< self._layers.len());
        self._layers[i].as_ref()
    }

    /// Forwards the input through all [Layer]s and returns the final result.
    pub fn forward(&self, inp: &CalcNode) -> CalcNode {
        self._layers
            .iter()
            .fold(inp.clone(), |out, layer| layer.forward(&out))
    }

    // Helps calculate the loss
    fn calc_regularization(&self) -> CalcNode {
        if let Some(regularization) = self._regularization {
            if regularization <= 0. {
                panic!("Regularization coefficient must be positive.");
            }
            let regularization = CalcNode::new_scalar(regularization);
            let n_param = self.param_iter().count();
            let n_param = CalcNode::new_scalar(n_param as FloatType);
            // Mean of the sum of the squares of all parameters
            let param = self.param_iter();
            param
                .map(|p| p.pow(&CalcNode::new_scalar(2.)).sum())
                .sum::<CalcNode>()
                * regularization
                / n_param
        } else {
            CalcNode::new_scalar(0.)
        }
    }

    /// Calculates the average loss of the list of (prediction, truth) tuples.
    pub fn loss(&self, inp: &[(CalcNode, CalcNode)]) -> CalcNode {
        let loss = inp
            .iter()
            .map(|(inp, truth)| (self._loss_func)(&self.forward(inp), truth))
            .sum::<CalcNode>()
            * CalcNode::new_scalar(1. / inp.len() as FloatType);
        let reg = self.calc_regularization();
        loss + reg
    }

    /// Trains the network on the supplied training data and returns the average loss.
    pub fn train(&mut self, inp: &[(CalcNode, CalcNode)], learning_rate: FloatType) -> FloatType {
        let mut loss = self.loss(inp);
        loss.back_propagation();
        self.decend_grad(learning_rate);

        loss.value_indexed(0)
    }

    /// Lets every parameter decend its respective gradient.
    fn decend_grad(&mut self, learning_rate: FloatType) {
        self.param_iter_mut()
            .for_each(|p| p.decend_grad(learning_rate));
    }

    /// Exports all parameters to a text file.
    /// 
    /// The filename is returned upon successful export.
    /// 
    /// Adds a numerical suffix if the wanted filename is taken.
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
        for (n, param) in self.param_iter().enumerate() {
            writeln!(file, "Parameter BEGIN: {n}")?;
            for i in 0..param.len() {
                writeln!(file, "{}", param.value_indexed(i))?;
            }
            writeln!(file, "Parameter END: {n}")?;
        }
        Ok(fn_string)
    }

    /// Imports previously exported parameters. Expect errors if the current network does not have the same setup of layers.
    pub fn import_parameters(&mut self, filename: &str) -> Result<(), Error> {
        let mut param_vals: Vec<FloatType> = Vec::new();
        let file_content = read_to_string(filename);
        match file_content {
            Ok(content) => {
                let mut imported_parameters = 0;
                let target_parameters = self.param_iter().count();
                let mut target_iter = self.param_iter_mut();
                for line in content.lines() {
                    if line.starts_with("Parameter BEGIN") {
                        // Do nothing
                    } else if line.starts_with("Parameter END") {
                        if let Some(target) = target_iter.next() {
                            imported_parameters += 1;
                            assert_eq!(
                                target.len(),
                                param_vals.len(),
                                "Wrong size of parameter {} from file.",
                                imported_parameters
                            );
                            target.set_vals(&param_vals);
                        }
                        param_vals.clear();
                    } else {
                        param_vals.push(line.parse().unwrap())
                    }
                }
                if imported_parameters < target_parameters {
                    eprintln!("Parameter file contained too few parameters, only the first {imported_parameters} were set.");
                }
                Ok(())
            }
            Err(err) => Err(err),
        }
    }
}

impl Display for MultiLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "MLP: [")?;
        for layer in &self._layers {
            layer.fmt(f)?;
        }
        writeln!(f, "]")
    }
}

impl Parameters for MultiLayer {
    fn param_iter(&self) -> Box<dyn Iterator<Item = &CalcNode> + '_> {
        Box::new(self._layers.iter().flat_map(|l| l.param_iter()))
    }
    fn param_iter_mut(&mut self) -> Box<dyn Iterator<Item = &mut CalcNode> + '_> {
        Box::new(self._layers.iter_mut().flat_map(|l| l.param_iter_mut()))
    }
}
