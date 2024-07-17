use rand::Rng;
use std::fs;

pub struct DataSet {
    the_data: String,
    training_len: usize,
}

impl DataSet {
    pub fn new(path: &str, training_ratio: f32, lowercase: bool) -> Self {
        let the_data = Self::get_string_from_file(path, lowercase);
        let training_len = Self::calc_training_len(the_data.len(), training_ratio);
        DataSet {
            the_data,
            training_len,
        }
    }

    fn calc_training_len(data_len: usize, ratio: f32) -> usize {
        (data_len as f32 * ratio) as usize
    }

    fn get_string_from_file(path: &str, lowercase: bool) -> String {
        let data =
            fs::read_to_string(path).and_then(|s| if lowercase { Ok(s.to_lowercase()) } else { Ok(s) });
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
        self.training_len = Self::calc_training_len(self.the_data.len(), ratio);
    }

    pub fn get_training_data(&self) -> &str {
        &self.the_data[..self.training_len]
    }

    pub fn get_validation_data(&self) -> &str {
        &self.the_data[self.training_len..]
    }

    pub fn get_training_block(&self, block_size: usize) -> &str {
        if block_size + 1 >= self.training_len {
            return self.get_training_data();
        }
        let end = rand::thread_rng().gen_range(block_size..=self.training_len);
        &self.the_data[end - block_size..end]
    }

    pub fn get_validation_block(&self, block_size: usize) -> &str {
        let validation_len = self.the_data.len() - self.training_len;
        if block_size <= validation_len {
            return self.get_validation_data();
        }
        let end = self.training_len + rand::thread_rng().gen_range(block_size..validation_len);
        &self.the_data[end - block_size..end]
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
        assert!(ds.the_data.starts_with("First Citizen:"));
        assert!(ds.the_data.ends_with("Whiles thou art waking.\n"));
    }

    #[test]
    fn finding_all_characters_in_shakespeare() {
        let ds = import_shakespeare();
        let charset = CharSet::from_str(ds.get_training_data()).unwrap();
        assert_eq!(charset.size(), 65);
    }
}
