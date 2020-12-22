use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::convert;
use std::fs;
use std::iter;
use unicode_segmentation::UnicodeSegmentation;

pub type Frequency = f64;
pub type Probability = f64;
pub type Count = u32;

#[derive(PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct BagOfWords {
    bow: HashMap<String, Count>,
}

impl BagOfWords {
    pub fn new() -> Self {
        BagOfWords {
            bow: HashMap::new(),
        }
    }

    pub fn from_file(file_path: &str) -> Option<Self> {
        fs::read_to_string(file_path)
            .ok()
            .and_then(|s| Some(BagOfWords::from(&s[..])))
    }

    pub fn from_folder(dir_path: &str) -> Self {
        fs::read_dir(dir_path)
            .expect("ok")
            .filter_map(|entry| {
                entry
                    .ok()
                    .and_then(|e| e.path().to_str().and_then(|p| BagOfWords::from_file(p)))
            })
            .collect()
    }

    pub fn combine(mut self, other: Self) -> Self {
        for (k, v) in other.bow {
            self.bow.entry(k).and_modify(|sv| *sv += v).or_insert(v);
        }
        self
    }

    pub fn total_word_count(&self) -> Count {
        self.bow.values().fold(0, |acc, n| acc + n)
    }

    pub fn word_frequency(&self, word: &str) -> Option<Frequency> {
        let word_vec: Vec<&str> = word
            .split_word_bounds()
            .filter(|&s| !s.trim().is_empty())
            .collect();
        if word_vec.len() == 0 || word_vec.len() > 1 {
            return None;
        }

        self.bow
            .get(word_vec[0])
            .and_then(|&v| Some(v as Frequency / self.total_word_count() as Frequency))
    }
}

impl convert::From<&str> for BagOfWords {
    fn from(s: &str) -> BagOfWords {
        let mut bow = BagOfWords::new();
        for w in s.split_word_bounds().filter(|&s| !s.trim().is_empty()) {
            *bow.bow.entry(w.to_uppercase()).or_insert(0) += 1;
        }
        bow
    }
}

impl iter::FromIterator<BagOfWords> for BagOfWords {
    fn from_iter<I: IntoIterator<Item = BagOfWords>>(iter: I) -> Self {
        let mut c = BagOfWords::new();
        for i in iter {
            c = c.combine(i);
        }
        c
    }
}

pub struct HSModel {
    ham_bow: BagOfWords,
    spam_bow: BagOfWords,
}

impl HSModel {
    pub fn new() -> Self {
        HSModel {
            ham_bow: BagOfWords::new(),
            spam_bow: BagOfWords::new(),
        }
    }

    pub fn add_spam_bow(mut self, spam_bow: BagOfWords) -> Self {
        self.spam_bow = self.spam_bow.combine(spam_bow);
        self
    }

    pub fn add_ham_bow(mut self, ham_bow: BagOfWords) -> Self {
        self.ham_bow = self.ham_bow.combine(ham_bow);
        self
    }

    pub fn from_bows(ham_bow: BagOfWords, spam_bow: BagOfWords) -> Self {
        HSModel::new().add_ham_bow(ham_bow).add_spam_bow(spam_bow)
    }

    pub fn text_spam_probability(&self, text: &str) -> Frequency {
        let (product, prod_of_diffs) = text
            .split_word_bounds()
            .filter(|&s| !s.trim().is_empty())
            .filter_map(|word| {
                if let (Some(spam_freq), Some(ham_freq)) = (
                    self.spam_bow.word_frequency(word),
                    self.ham_bow.word_frequency(word),
                ) {
                    Some(spam_freq / (spam_freq * ham_freq))
                } else {
                    None
                }
            })
            .fold((1f64, 1f64), |acc, p| (acc.0 * p, acc.1 * (1f64 - p)));
        product / (product + prod_of_diffs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /*****************************************/
    /* FROM &str TESTS */
    /*****************************************/

    #[test]
    fn new_bow() {
        let fbow: BagOfWords = BagOfWords::new();
        let bow = BagOfWords {
            bow: HashMap::new(),
        };
        assert_eq!(fbow, bow);
    }

    #[test]
    fn bow_from_empty_string() {
        let fbow: BagOfWords = BagOfWords::from("");
        let bow = BagOfWords::new();
        assert_eq!(fbow, bow);
    }

    #[test]
    fn bow_from_one_word() {
        let fbow: BagOfWords = BagOfWords::from("hello");
        let bow = BagOfWords {
            bow: {
                let mut hm = HashMap::new();
                hm.insert("HELLO".to_string(), 1u32);
                hm
            },
        };
        assert_eq!(fbow, bow);
    }
    #[test]
    fn bow_from_2_eq_words() {
        let fbow: BagOfWords = BagOfWords::from("hElLo hello");
        let bow = BagOfWords {
            bow: {
                let mut hm = HashMap::new();
                hm.insert("HELLO".to_string(), 2u32);
                hm
            },
        };
        assert_eq!(fbow, bow);
    }

    #[test]
    fn bow_from_unicode() {
        let fbow: BagOfWords = BagOfWords::from("😊");
        let bow = BagOfWords {
            bow: {
                let mut hm = HashMap::new();
                hm.insert("😊".to_string(), 1u32);
                hm
            },
        };
        assert_eq!(fbow, bow);
    }

    #[test]
    fn bow_2_from_unicode() {
        let fbow: BagOfWords = BagOfWords::from("😊 😊");
        let bow = BagOfWords {
            bow: {
                let mut hm = HashMap::new();
                hm.insert("😊".to_string(), 2u32);
                hm
            },
        };
        assert_eq!(fbow, bow);
    }

    #[test]
    fn bow_2_from_unicode_no_spaces_emoji() {
        let fbow: BagOfWords = BagOfWords::from("😊hello😊");
        let bow = BagOfWords {
            bow: {
                let mut hm = HashMap::new();
                hm.insert("😊".to_string(), 2u32);
                hm.insert("HELLO".to_string(), 1u32);
                hm
            },
        };
        assert_eq!(fbow, bow);
    }

    #[test]
    fn bow_from_2_emoji_no_space() {
        let fbow: BagOfWords = BagOfWords::from("😊😊");
        let bow = BagOfWords {
            bow: {
                let mut hm = HashMap::new();
                hm.insert("😊".to_string(), 2u32);
                hm
            },
        };
        assert_eq!(fbow, bow);
    }

    #[test]
    #[ignore] //ignoring unless I think this is necessary
    fn bow_from_ascii_skip_punctuation() {
        let fbow: BagOfWords = BagOfWords::from("hi. there. you.");
        let bow = BagOfWords {
            bow: {
                let mut hm = HashMap::new();
                hm.insert("HI".to_string(), 1u32);
                hm.insert("HI".to_string(), 1u32);
                hm.insert("HI".to_string(), 1u32);
                hm
            },
        };
        assert_eq!(fbow, bow);
    }

    /*****************************************/
    /* COMBINE TESTS                         */
    /*****************************************/

    #[test]
    fn combine_empty_bows() {
        let fbow = BagOfWords::combine(BagOfWords::from(""), BagOfWords::from(""));
        let bow = BagOfWords::new();
        assert_eq!(fbow, bow);
    }

    #[test]
    fn combine_non_empty_with_empty() {
        let fbow = BagOfWords::combine(BagOfWords::from("HELLO"), BagOfWords::from(""));
        let bow = BagOfWords::from("HELLO");
        assert_eq!(fbow, bow);
    }

    #[test]
    fn combine_empty_with_non_empty() {
        let fbow = BagOfWords::combine(BagOfWords::from(""), BagOfWords::from("HELLO"));
        let bow = BagOfWords::from("HELLO");
        assert_eq!(fbow, bow);
    }

    #[test]
    fn combine_both_non_empty() {
        let fbow = BagOfWords::combine(BagOfWords::from("HELLO"), BagOfWords::from("HELLO"));
        let bow = BagOfWords::from("HELLO HELLO");
        assert_eq!(fbow, bow);
    }

    #[test]
    fn combine_both_non_empty_different() {
        let fbow = BagOfWords::combine(
            BagOfWords::from("HELLO there beautiful world"),
            BagOfWords::from("HELLO"),
        );
        let bow = BagOfWords::from("HELLO there beautiful world hello");
        assert_eq!(fbow, bow);
    }

    #[test]
    fn combine_three() {
        let fbow = BagOfWords::new()
            .combine(BagOfWords::from("hello there world"))
            .combine(BagOfWords::from("hello there world 😊😊😊😊😊"))
            .combine(BagOfWords::from("😊😊😊😊😊"));
        let bow: BagOfWords =
            BagOfWords::from("hello there world hello there world 😊😊😊😊😊😊😊😊😊😊");
        assert_eq!(fbow, bow)
    }

    /*****************************************/
    /* FROM ITER TESTS                         */
    /*****************************************/

    #[test]
    fn from_iter() {
        let bowvec: Vec<BagOfWords> = vec![
            BagOfWords::from("hello there world"),
            BagOfWords::from("hello there world 😊😊😊😊😊"),
            BagOfWords::from("😊😊😊😊😊"),
        ];

        let fbow: BagOfWords = bowvec.into_iter().collect();
        let bow: BagOfWords =
            BagOfWords::from("hello there world hello there world 😊😊😊😊😊😊😊😊😊😊");
        assert_eq!(fbow, bow)
    }

    /*****************************************/
    /* FROM FILE TESTS                     */
    /*****************************************/

    #[test]
    fn bow_from_file_ascii_only() {
        let fbow: BagOfWords = BagOfWords::from_file("test_data/ascii_only.txt").unwrap();
        let bow = BagOfWords::from("HELLO THERE WORLD");
        assert_eq!(fbow, bow);
    }

    #[test]
    fn bow_from_file_unicode_only() {
        let fbow: BagOfWords = BagOfWords::from_file("test_data/unicode_only.txt").unwrap();
        let bow = BagOfWords::from("😊😊😊😊😊");
        assert_eq!(fbow, bow);
    }

    #[test]
    fn bow_from_file_unicode_and_ascii() {
        let fbow: BagOfWords = BagOfWords::from_file("test_data/unicode_and_ascii.txt").unwrap();
        let bow = BagOfWords::from("😊😊😊😊😊 HELLO THERE WORLD");
        assert_eq!(fbow, bow);
    }

    /*****************************************/
    /* FROM FOLDER TESTS                     */
    /*****************************************/

    #[test]
    fn bow_from_test_data_folder() {
        let fbow: BagOfWords = BagOfWords::from_folder("test_data");
        let bow = BagOfWords::new()
            .combine(BagOfWords::from("hello there world"))
            .combine(BagOfWords::from("hello there world 😊😊😊😊😊"))
            .combine(BagOfWords::from("😊😊😊😊😊"));

        assert_eq!(fbow, bow);
    }
}
