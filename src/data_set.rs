use rand::Rng;
use std::fs;

pub struct DataSet {
    _the_data: String,
    _training_len: usize,
}

impl DataSet {
    pub fn new(path: &str, training_ratio: f32, lowercase: bool) -> Self {
        let the_data = Self::get_string_from_file(path, lowercase);
        let training_len = Self::calc_training_len(the_data.len(), training_ratio);
        DataSet {
            _the_data: the_data,
            _training_len: training_len,
        }
    }

    fn calc_training_len(data_len: usize, ratio: f32) -> usize {
        (data_len as f32 * ratio) as usize
    }

    fn get_string_from_file(path: &str, lowercase: bool) -> String {
        let data =
            fs::read_to_string(path).map(|s| if lowercase { s.to_lowercase() } else { s });
        if data.is_err() {
            panic!(
                "Received {:?}  while importing dataset from {}.",
                data.err(),
                path
            );
        }
        data.unwrap()
    }

    pub fn set_training_ratio(&mut self, ratio: f32) {
        self._training_len = Self::calc_training_len(self._the_data.len(), ratio);
    }

    pub fn training_data(&self) -> &str {
        &self._the_data[..self._training_len]
    }

    pub fn training_len(&self) -> usize {
        self._training_len
    }

    pub fn validation_data(&self) -> &str {
        &self._the_data[self._training_len..]
    }

    pub fn validation_len(&self) -> usize {
        self._the_data.len() - self._training_len
    }

    pub fn training_block(&self, block_size: usize) -> &str {
        if block_size + 1 >= self._training_len {
            return self.training_data();
        }
        let end = rand::thread_rng().gen_range(block_size..=self._training_len);
        &self._the_data[end - block_size..end]
    }

    pub fn validation_block(&self, block_size: usize) -> &str {
        let validation_len = self._the_data.len() - self._training_len;
        if block_size <= validation_len {
            return self.validation_data();
        }
        let end = self._training_len + rand::thread_rng().gen_range(block_size..validation_len);
        &self._the_data[end - block_size..end]
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use crate::nnetwork::CharSet;

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
        assert!(ds._the_data.starts_with("First Citizen:"));
        assert!(ds._the_data.ends_with("Whiles thou art waking.\n"));
    }

    #[test]
    fn finding_all_characters_in_shakespeare() {
        let ds = import_shakespeare();
        let charset = CharSet::from_str(ds.training_data()).unwrap();
        assert_eq!(charset.size(), 64);
    }
}
