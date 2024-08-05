use std::fs;

use crate::nnetwork::{FloatType, TensorShared, TensorType, VecOrientation};

#[derive(Debug, PartialEq)]
pub enum DataSetError {
    Encoding(char),
    DecodingVector(Vec<FloatType>),
    DecodingIndex(usize),
    Creation,
}

pub struct DataSet {
    _data: String,
    _chars: Vec<char>,
    _training_data: Vec<String>,
    _validation_data: Vec<String>,
}

impl DataSet {
    pub fn new(path: &str, training_ratio: f32, lowercase: bool) -> Self {
        let data = Self::get_string_from_file(path, lowercase);
        let mut training_data = Vec::new();
        let mut validation_data = Vec::new();
        let mut n_newlines = (data.lines().count() as f32 * training_ratio) as usize - 1;
        for line in data.lines() {
            if n_newlines > 0 {
                training_data.push(line.to_string());
                n_newlines -= 1;
            } else {
                validation_data.push(line.to_string());
            }
        }

        let mut chars = Vec::new();
        data.chars().for_each(|c: char| {
            if c.is_ascii_alphabetic() && !chars.contains(&c) {
                chars.push(c);
            }
        });
        chars.sort();

        DataSet {
            _data: data,
            _chars: chars,
            _training_data: training_data,
            _validation_data: validation_data,
        }
    }

    pub fn add_character(&mut self, c: char) {
        if !self._chars.contains(&c) {
            self._chars.push(c);
        }
    }

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

    pub fn training_data(&self) -> &[String] {
        &self._training_data
    }

    pub fn validation_data(&self) -> &[String] {
        &self._validation_data
    }

    pub fn number_of_chars(&self) -> usize {
        self._chars.len()
    }

    pub fn decode(&self, vector: &TensorShared) -> Result<char, DataSetError> {
        if vector.tensor_type() != TensorType::Vector(VecOrientation::Column) {
            panic!("Can only decode column vectors.");
        }
        let index: Vec<usize> = vector
            .value()
            .iter()
            .enumerate()
            .filter_map(|(n, &elem)| if elem > 0. { Some(n) } else { None })
            .collect();
        if index.len() != 1 {
            return Err(DataSetError::DecodingVector(vector.value().to_vec()));
        }
        let index = index[0];
        Ok(self
            ._chars
            .get(index)
            .ok_or(DataSetError::DecodingIndex(index))?)
        .copied()
    }

    pub fn decode_string(&self, v: &[&TensorShared]) -> Result<String, DataSetError> {
        v.iter().map(|v| self.decode(v)).collect()
    }

    pub fn encode(&self, s: &str) -> Result<TensorShared, DataSetError> {
        let n_rows = self._chars.len();
        let n_cols = s.len();
        let mut out = TensorShared::from_vector(vec![0.; n_rows * n_cols], (n_rows, n_cols, 1));
        for (col, ch) in s.chars().enumerate() {
            if let Some(row) = self._chars.iter().position(|&k| ch == k) {
                out.set_index(row, col, 0, 1.);
            } else {
                return Err(DataSetError::Encoding(ch));
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn import_shakespeare() -> DataSet {
        DataSet::new("./datasets/tiny_shakespeare.txt", 0.9, false)
    }

    #[test]
    #[should_panic]
    fn wrong_path() {
        DataSet::new("no dataset here", 0.9, false);
    }

    #[test]
    fn reading_all_shakespeare() {
        let ds = import_shakespeare();
        assert!(ds._data.starts_with("First Citizen:"));
        assert!(ds._data.ends_with("Whiles thou art waking.\n"));
    }

    #[test]
    fn finding_all_characters_in_shakespeare() {
        let ds = import_shakespeare();
        assert_eq!(ds.number_of_chars(), 64);
    }
}
