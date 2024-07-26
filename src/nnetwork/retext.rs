use std::str::FromStr;

use crate::data_preparing::{char_set::CharSet, data_set::DataSet};

use super::{FunctionLayer, LinearLayer, MultiLayer};

pub struct ReText {
    _data: DataSet,
    _chars: CharSet,
    _ml: MultiLayer,
}

impl ReText {
    pub fn new(
        data: DataSet,
        data_block_size: usize,
        embed_dim: usize,
        n_hidden_layers: usize,
        layer_dim: usize,
    ) -> ReText {
        let chars = CharSet::from_str(data.training_data()).unwrap();
        let n_chars = chars.len();
        let mut ml = MultiLayer::from_empty();
        let non_linearity = FunctionLayer::new(&FunctionLayer::tanh, "Tanh");
        const BIASED_LAYERS: bool = true;

        // Embedding layer
        let embed = LinearLayer::from_rand(
            data_block_size * n_chars,
            data_block_size * embed_dim,
            false,
        );
        ml.add_layer(Box::new(embed));
        ml.add_layer(Box::new(LinearLayer::from_rand(
            data_block_size * embed_dim,
            layer_dim,
            BIASED_LAYERS,
        )));
        ml.add_layer(Box::new(non_linearity.clone()));

        // Hidden layers
        for _ in 0..n_hidden_layers {
            ml.add_layer(Box::new(LinearLayer::from_rand(
                layer_dim,
                layer_dim,
                BIASED_LAYERS,
            )));
            ml.add_layer(Box::new(non_linearity.clone()));
        }

        // Deembed
        ml.add_layer(Box::new(LinearLayer::from_rand(
            layer_dim,
            n_chars,
            BIASED_LAYERS,
        )));
        ml.add_layer(Box::new(non_linearity.clone()));

        ml.add_layer(Box::new(FunctionLayer::new(
            &FunctionLayer::softmax,
            "SoftMax",
        )));

        ReText {
            _data: data,
            _chars: chars,
            _ml: ml,
        }
    }

    pub fn train() {}
    
    fn extract_correlations() {}
    
    pub fn predict() {}
}
