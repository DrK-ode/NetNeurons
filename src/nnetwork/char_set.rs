use std::{fmt::Display, str::FromStr};

use super::{GradVal, GradValVec};

#[derive(Debug, PartialEq)]
pub enum CharSetError {
    EncodingError(char),
    DecodingVectorError(Vec<f32>),
    DecodingIndexError(usize),
    CreationError,
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

    pub fn from_str_vec<T>(vec: &Vec<T>) -> Self
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

    pub fn decode(&self, vector: &GradValVec) -> Result<char, CharSetError> {
        let index: Vec<usize> = vector
            //.column(0)
            .iter()
            .enumerate()
            .filter_map(|(n, elem)| if elem.value() > 0. { Some(n) } else { None })
            .collect();
        if index.len() != 1 {
            return Err(CharSetError::DecodingVectorError(
                vector.iter().map(|v| v.value()).collect(),
            ));
        }
        let index = index[0];
        Ok(self
            ._characters
            .get(index)
            .ok_or_else(|| CharSetError::DecodingIndexError(index))?)
        .copied()
    }

    pub fn decode_string(&self, v: Vec<GradValVec>) -> Result<String, CharSetError> {
        v.iter().map(|v| self.decode(v)).collect()
    }

    pub fn encode(&self, c: char) -> Result<GradValVec, CharSetError> {
        let n = self
            ._characters
            .iter()
            .position(|k| c == *k)
            .ok_or_else(|| CharSetError::EncodingError(c))?;
        let mut vector = vec![GradVal::from(0.); self._characters.len()];
        vector[n] = GradVal::from(1.);
        Ok(vector.into())
    }

    pub fn encode_string(&self, s: &str) -> Result<Vec<GradValVec>, CharSetError> {
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
            CharSet::from_str("abc").unwrap().encode('a').unwrap(),
            GradValVec::from(vec![1., 0., 0.])
        );
    }

    #[test]
    fn encode_c() {
        assert_eq!(
            CharSet::from_str("abc").unwrap().encode('c').unwrap(),
            GradValVec::from(vec![0., 0., 1.])
        );
    }

    #[test]
    fn characters_are_sorted() {
        assert_eq!(
            CharSet::from_str("cab")
                .unwrap()
                .encode_string("abc")
                .unwrap(),
            vec!(
                GradValVec::from(vec![1., 0., 0.]),
                GradValVec::from(vec![0., 1., 0.]),
                GradValVec::from(vec![0., 0., 1.])
            )
        );
    }

    #[test]
    fn ambigious_decode() {
        assert_eq!(
            CharSet::from_str("abc").unwrap().decode(&vec![1., 1., 0.].into()),
            Err(CharSetError::DecodingVectorError(vec![1., 1., 0.]))
        );
    }

    #[test]
    fn decode_a() {
        assert_eq!(
            CharSet::from_str("abc")
                .unwrap()
                .decode(&vec![1., 0., 0.].into())
                .unwrap(),
            'a'
        );
    }

    #[test]
    fn decode_c() {
        assert_eq!(
            CharSet::from_str("abc")
                .unwrap()
                .decode(&vec![0., 0., 1.].into())
                .unwrap(),
            'c'
        );
    }

    #[test]
    fn decode_abc() {
        assert_eq!(
            CharSet::from_str("abc")
                .unwrap()
                .decode_string(vec!(vec![1., 0., 0.].into(), vec![0., 1., 0.].into(), vec![0., 0., 1.].into()))
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
                .decode_string(vec!(vec![1., 0., 0.].into(), vec![0., 1., 0.].into(), vec![0., 0., 1.].into()))
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
                .decode_string(vec!(vec![1., 0., 0.].into(), vec![0., 1., 0.].into(), vec![0., 0., 1.].into()))
                .unwrap(),
            "abc"
        );
    }
}
