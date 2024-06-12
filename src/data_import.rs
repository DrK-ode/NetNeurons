use std::fs;

fn import_dataset(path: &str) -> String {
    let data = fs::read_to_string(path);
    if data.is_err() {
        panic!("Received {:?}  while importing dataset from {}.", data.err(), path);
    }
    data.unwrap()
}

pub fn import_names(training_ratio: f32) -> (Vec<String>, Vec<String>) {
    let names = import_dataset("./datasets/names.txt");
    let n = names.lines().count() as f32;
    let n_training = (training_ratio * n) as usize;
    let data_training = names
        .lines()
        .take(n_training)
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    let data_validate = names
        .lines()
        .skip(n_training)
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    (data_training, data_validate)
}

pub fn import_shakespeare(training_ratio: f32) -> (String, String) {
    let mut data_training = import_dataset("./datasets/tiny_shakespeare.txt");
    let data_validate =
        data_training.split_off((data_training.len() as f32 * training_ratio) as usize);
    (data_training, data_validate)
}

#[cfg(test)]
mod tests{
    use std::str::FromStr;

    use crate::bigram::CharSet;

    use super::*;

    #[test]
    #[should_panic]
    fn wrong_path(){
        import_dataset("no dataset here");
    }

    #[test]
    fn reading_all_names(){
        let (a,b) = import_names(0.9);
        assert_eq!(a.len()+b.len(), 32033);
    }

    #[test]
    fn correct_training_ratio(){
        let (a,b) = import_names(0.9);
        assert_eq!(a.len(), 28829);
        assert_eq!(b.len(), 3204);
    }

    #[test]
    fn reading_all_shakespeare(){
        let (a,b) = import_shakespeare(0.9);
        assert!(a.starts_with("First Citizen:"));
        assert!(b.ends_with("Whiles thou art waking.\n"));
    }

    #[test]
    fn finding_all_characters_in_names(){
        let (a,_) = import_names(0.9);
        let charset = CharSet::from_str_vec(&a);
        assert_eq!(charset.size(), 26);
    }

    #[test]
    fn finding_all_characters_in_shakespeare(){
        let (a,_) = import_shakespeare(0.9);
        let charset = CharSet::from_str(&a).unwrap();
        assert_eq!(charset.size(), 65);
    }
}
