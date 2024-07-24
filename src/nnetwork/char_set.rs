use std::{fmt::Display, str::FromStr};

use super::calculation_nodes::{FloatType, TensorShared, TensorType, VecOrientation};

#[derive(Debug, PartialEq)]
pub enum CharSetError {
    Encoding(char),
    DecodingVector(Vec<FloatType>),
    DecodingIndex(usize),
    Creation,
}

#[derive(Debug, Default, PartialEq)]
pub struct CharSet {
    _characters: Vec<char>,
}

impl CharSet {
    // Added characters after the initial construction will be added last, the characters are not automatically sorted
    pub fn add_character(&mut self, c: char) -> &mut Self {
        if !self._characters.contains(&c) {
            self._characters.push(c);
        }
        self
    }

    pub fn sort(&mut self) -> &mut Self {
        self._characters.sort();
        self
    }

    pub fn from_str_vec<T>(vec: &[T]) -> Self
    where
        T: ToString,
    {
        let mut charset = CharSet::default();
        vec.iter().for_each(|s| {
            s.to_string().chars().for_each(|c| {
                charset.add_character(c);
            })
        });
        charset._characters.sort();
        charset
    }

    pub fn size(&self) -> usize {
        self._characters.len()
    }

    pub fn decode(&self, vector: &TensorShared) -> Result<char, CharSetError> {
        if vector.tensor_type() != TensorType::Vector(VecOrientation::Column) {
            panic!("Can only decode column vectors.");
        }
        let index: Vec<usize> = vector
            .value()
            //.column(0)
            .iter()
            .enumerate()
            .filter_map(|(n, &elem)| if elem > 0. { Some(n) } else { None })
            .collect();
        if index.len() != 1 {
            return Err(CharSetError::DecodingVector(
                vector.value().to_vec(),
            ));
        }
        let index = index[0];
        Ok(self
            ._characters
            .get(index)
            .ok_or(CharSetError::DecodingIndex(index))?)
        .copied()
    }

    pub fn decode_string(&self, v: &[&TensorShared]) -> Result<String, CharSetError> {
        v.iter().map(|v| self.decode(v)).collect()
    }

    pub fn encode(&self, c: char) -> Result<TensorShared, CharSetError> {
        let n = self
            ._characters
            .iter()
            .position(|k| c == *k)
            .ok_or(CharSetError::Encoding(c))?;
        let size = self._characters.len();
        let mut vector = vec![0.; size];
        vector[n] = 1.;
        Ok(TensorShared::from_vector(vector, (size, 1, 1)))
    }

    pub fn encode_string(&self, s: &str) -> Result<Vec<TensorShared>, CharSetError> {
        s.chars().map(|c| self.encode(c)).collect()
    }
}

impl Display for CharSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            self._characters
                .iter()
                .collect::<String>()
                .replace('\n', "\\n")
                .replace(' ', "\\ ")
                .replace('\t', "\\t")
        )
    }
}

impl FromStr for CharSet {
    type Err = CharSetError;

    fn from_str(data: &str) -> Result<CharSet, CharSetError> {
        let mut charset = CharSet::default();
        data.chars().for_each(|c| {
            charset.add_character(c);
        });
        charset._characters.sort_unstable();
        Ok(charset)
    }
}

impl FromIterator<char> for CharSet {
    fn from_iter<T: IntoIterator<Item = char>>(iter: T) -> Self {
        let mut charset = CharSet::default();
        iter.into_iter().for_each(|c| {
            charset.add_character(c);
        });
        charset._characters.sort();
        charset
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construct_from_str() {
        assert_eq!(CharSet::from_str("abcdefgh").unwrap().size(), 8);
    }

    #[test]
    fn encode_a() {
        assert_eq!(
            CharSet::from_str("abc")
                .unwrap()
                .encode('a')
                .unwrap()
                .value_as_col_vector()
                .unwrap(),
            &[1., 0., 0.]
        );
    }

    #[test]
    fn encode_c() {
        assert_eq!(
            CharSet::from_str("abc")
                .unwrap()
                .encode('c')
                .unwrap()
                .value_as_col_vector()
                .unwrap(),
            &[0., 0., 1.]
        );
    }

    #[test]
    fn characters_are_sorted() {
        assert_eq!(
            CharSet::from_str("cab")
                .unwrap()
                .encode_string("abc")
                .unwrap()
                .iter()
                .map(|v| v.value_as_col_vector().unwrap())
                .collect::<Vec<_>>(),
            vec![&[1., 0., 0.], &[0., 1., 0.], &[0., 0., 1.]]
        );
    }

    #[test]
    fn ambigious_decode() {
        assert_eq!(
            CharSet::from_str("abc")
                .unwrap()
                .decode(&TensorShared::from_vector(vec![1., 1., 0.], (3, 1, 1))),
            Err(CharSetError::DecodingVector(vec![1., 1., 0.]))
        );
    }

    #[test]
    fn decode_a() {
        assert_eq!(
            CharSet::from_str("abc")
                .unwrap()
                .decode(&TensorShared::from_vector(vec![1., 0., 0.], (3, 1, 1)))
                .unwrap(),
            'a'
        );
    }

    #[test]
    fn decode_c() {
        assert_eq!(
            CharSet::from_str("abc")
                .unwrap()
                .decode(&TensorShared::from_vector(vec![0., 0., 1.], (3, 1, 1)))
                .unwrap(),
            'c'
        );
    }

    #[test]
    fn decode_abc() {
        assert_eq!(
            CharSet::from_str("abc")
                .unwrap()
                .decode_string(&[
                    &TensorShared::from_vector(vec![1., 0., 0.], (3, 1, 1)),
                    &TensorShared::from_vector(vec![0., 1., 0.], (3, 1, 1)),
                    &TensorShared::from_vector(vec![0., 0., 1.], (3, 1, 1))
                ])
                .unwrap(),
            "abc"
        );
    }

    #[test]
    fn adding_characters() {
        assert_eq!(
            CharSet::from_str("bc")
                .unwrap()
                .add_character('a')
                .decode_string(&[
                    &TensorShared::from_vector(vec![1., 0., 0.], (3, 1, 1)),
                    &TensorShared::from_vector(vec![0., 1., 0.], (3, 1, 1)),
                    &TensorShared::from_vector(vec![0., 0., 1.], (3, 1, 1))
                ])
                .unwrap(),
            "bca"
        );
    }

    #[test]
    fn sorting_charset() {
        assert_eq!(
            CharSet::from_str("bc")
                .unwrap()
                .add_character('a')
                .sort()
                .decode_string(&[
                    &TensorShared::from_vector(vec![1., 0., 0.], (3, 1, 1)),
                    &TensorShared::from_vector(vec![0., 1., 0.], (3, 1, 1)),
                    &TensorShared::from_vector(vec![0., 0., 1.], (3, 1, 1))
                ])
                .unwrap(),
            "abc"
        );
    }
}
