use std::fs;

use crate::nnetwork::{CalcNode, FloatType, NodeType, VecOrientation};

#[derive(Debug, PartialEq)]
pub enum DataSetError {
    Encoding(char),
    DecodingVector(Vec<FloatType>),
    DecodingIndex(usize),
    Creation,
}

/// Holds the data consisting of lines of text. Also holds an ordered set of all characters by which the [CharSet] encodes and decodes characters to and from [CalcNode]s.
pub struct CharSet {
    _data: String,
    _chars: Vec<char>,
    _training_data: Vec<String>,
    _validation_data: Vec<String>,
}

impl CharSet {
    /// Loads the data from the text file and creates the character set. Optionally all characters are made lowercase. The training ratio decides how much of the data that is available for training and not reserved for validation.
    /// 
    /// # Example
    /// ```
    /// use net_neurons::retext::CharSet;
    /// 
    /// let ds1 = CharSet::new("./datasets/tiny_shakespeare.txt", 1., true);
    /// assert_eq!(ds1.number_of_chars(), 26);
    /// let ds2 = CharSet::new("./datasets/tiny_shakespeare.txt", 1., false);
    /// assert_eq!(ds2.number_of_chars(), 52);
    /// ```
    pub fn new(path: &str, training_ratio: f32, lowercase: bool) -> Self {
        let data = Self::get_string_from_file(path, lowercase);
        let mut training_data = Vec::new();
        let mut validation_data = Vec::new();
        let n_training = (data.lines().count() as f32 * training_ratio) as usize - 1;

        data.lines()
            .take(n_training)
            .for_each(|line| training_data.push(line.to_string()));
        data.lines()
            .skip(n_training)
            .for_each(|line| validation_data.push(line.to_string()));

        let mut chars = Vec::new();
        data.chars().for_each(|c: char| {
            if c.is_ascii_alphabetic() && !chars.contains(&c) {
                chars.push(c);
            }
        });
        chars.sort();

        CharSet {
            _data: data,
            _chars: chars,
            _training_data: training_data,
            _validation_data: validation_data,
        }
    }

    /// Add extra characters to the set, e.g., sentinels.
    pub fn add_character(&mut self, c: char) {
        if !self._chars.contains(&c) {
            self._chars.push(c);
        }
    }

    /// Returns a slice of all currently known characters in the set.
    /// 
    /// # Example
    /// ```
    /// use net_neurons::retext::CharSet;
    /// 
    /// let ds = CharSet::new("./datasets/test.txt", 1., true);
    /// assert_eq!(ds.characters(), &['a','b','c']);
    /// ```
    pub fn characters(&self) -> &[char] {
        &self._chars
    }

    fn get_string_from_file(path: &str, lowercase: bool) -> String {
        let data = fs::read_to_string(path).map(|s| if lowercase { s.to_lowercase() } else { s });
        if data.is_err() {
            panic!(
                "Received {:?}  while importing dataset from {}.",
                data.err(),
                path
            );
        }
        data.unwrap()
    }

    /// Returns a slice of all available training data.
    pub fn training_data(&self) -> &[String] {
        &self._training_data
    }

    /// Returns a slice of all available validation data.
    pub fn validation_data(&self) -> &[String] {
        &self._validation_data
    }

    /// Number of known characters.
    pub fn number_of_chars(&self) -> usize {
        self._chars.len()
    }

    /// Interprets a [CalcNode] as a character. The [CalcNode] is assumed to be a one-hot vector.
    /// 
    /// # Example
    /// ```
    /// use net_neurons::{nnetwork::CalcNode,retext::CharSet};
    /// 
    /// let ds = CharSet::new("./datasets/test.txt", 1., true);
    /// let cn = CalcNode::new_col_vector(vec![0., 0., 1.]);
    /// assert_eq!( ds.decode_char(&cn), Ok('c') );
    /// ```
    pub fn decode_char(&self, vector: &CalcNode) -> Result<char, DataSetError> {
        if vector.node_type() != NodeType::Vector(VecOrientation::Column) {
            panic!("Can only decode column vectors.");
        }
        let index: Vec<usize> = vector
            .borrow()
            .vals()
            .iter()
            .enumerate()
            .filter_map(|(n, &elem)| if elem > 0. { Some(n) } else { None })
            .collect();
        if index.len() != 1 {
            return Err(DataSetError::DecodingVector(vector.copy_vals()));
        }
        let index = index[0];
        Ok(self
            ._chars
            .get(index)
            .ok_or(DataSetError::DecodingIndex(index))?)
        .copied()
    }

    /// Calls [CharSet::decode_char] for every [CalcNode]] in the slice, and returns a string. The [CalcNode]s are assumed to be a one-hot vectors.
    pub fn decode_string(&self, v: &[&CalcNode]) -> Result<String, DataSetError> {
        v.iter().map(|v| self.decode_char(v)).collect()
    }

    /// Creates a matrix where each column is a one-hot vector which corresponds a character in the string.
    /// 
    /// # Example
    /// ```
    /// use net_neurons::{nnetwork::CalcNode,retext::CharSet};
    /// 
    /// let ds = CharSet::new("./datasets/test.txt", 1., true);
    /// let cn = ds.encode("b").unwrap();
    /// assert_eq!(cn.value_indexed(0),0.);
    /// assert_eq!(cn.value_indexed(1),1.);
    /// assert_eq!(cn.value_indexed(2),0.);
    /// ```
    pub fn encode(&self, s: &str) -> Result<CalcNode, DataSetError> {
        let n_rows = self._chars.len();
        let n_cols = s.len();
        let mut out_vec = vec![0.; n_rows * n_cols];
        for (col, ch) in s.chars().enumerate() {
            if let Some(row) = self._chars.iter().position(|&k| ch == k) {
                out_vec[row * n_cols + col] = 1.;
            } else {
                return Err(DataSetError::Encoding(ch));
            }
        }
        Ok(CalcNode::new_from_shape((n_rows, n_cols), out_vec))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic]
    fn wrong_path() {
        CharSet::new("no dataset here", 0.9, false);
    }

    #[test]
    fn reading_all_shakespeare() {
        let ds = CharSet::new("./datasets/tiny_shakespeare.txt", 1., false);
        assert!(ds._data.starts_with("First Citizen:"));
        assert!(ds._data.ends_with("Whiles thou art waking.\n"));
    }

    #[test]
    fn finding_all_characters_in_shakespeare() {
        let ds = CharSet::new("./datasets/tiny_shakespeare.txt", 1., true);
        assert_eq!(ds.number_of_chars(), 26);
    }
}
