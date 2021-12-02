//! BagOfWords is used for representing frequency of word occurences in known spam/ham text.
//! HSModel uses a Spam BagOfWords and a Ham BagOfWords to calculate probability that a given text
//! is spam.
//! ```no_run
//! use rammer::{HSModel, BagOfWords};
//! let spam_bow = BagOfWords::from_folder("data/train/spam").expect("Folder not found");
//! let ham_bow = BagOfWords::from_folder("data/train/ham").expect("Folder not found");
//! let model = HSModel::from_bows(ham_bow, spam_bow);
//! model.text_spam_probability("hello i have an offer for you");
//! model.text_spam_probability("Hey it's greg, finished the data analysis");
//! ```  
use std::{collections::HashMap, convert, fs, iter};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use unicode_segmentation::UnicodeSegmentation;

use crate::{Count, Frequency};

/// A BagOfWords, also referred to as a bow, is a frequency map of words.
/// Read more about the BagOfWords model here: [BagOfWords Wikipedia](https://en.wikipedia.org/wiki/Bag-of-words_model).
/// BagOfWords works with Unicode Words. Words are defined by as between
/// [UAX#29 word boundaries](http://www.unicode.org/reports/tr29/#Word_Boundaries).
/// BagOfWords is serializable using one of the [serde serialization crates](https://serde.rs/#data-formats)
/// ```no_run
/// use rammer::BagOfWords;
/// use serde_json;
/// let singly_trained_bow = BagOfWords::from_file("test_resources/test_data/unicode_and_ascii.txt").expect("File not found");
/// let big_bow = BagOfWords::from_folder("data/train/ham").expect("Folder not found");
/// let com_bow = singly_trained_bow.combine(big_bow);
/// ```
#[derive(PartialEq, Eq, Debug, Serialize, Deserialize, Clone)]
pub struct BagOfWords {
    pub bow: HashMap<String, Count>,
}

#[allow(missing_doc_code_examples)]
impl BagOfWords {
    /// Return a new BagOfWords with an empty Frequency Map.
    /// ```
    /// # use rammer::BagOfWords;
    /// let empty_bow = BagOfWords::new();
    /// ```
    pub fn new() -> Self {
        BagOfWords {
            bow: HashMap::new(),
        }
    }

    /// Create a BagOfWords from a text file.
    /// This file should already be known to be ham or spam.
    /// The text file will be the basis of a new [HSModel's](struct.HSModel.html) Ham/Spam BagOfWords
    /// ```
    /// # use rammer::BagOfWords;
    /// let spam_bow = BagOfWords::from_file("test_resources/test_data/unicode_and_ascii.txt").unwrap();
    /// ```
    pub fn from_file(file_path: &str) -> Option<Self> {
        fs::read_to_string(file_path)
            .ok()
            .and_then(|s| Some(BagOfWords::from(&s[..])))
    }

    pub fn top_10_count(&self) -> Vec<(u32, String)> {
        let mut top_ten = vec![];
        for (word, count) in &self.bow {
            top_ten.push((count.clone(), word.to_string()));
        }

        top_ten.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        return top_ten[0..50].to_vec();
    }

    /// Create a BagOfWords from a folder containing either spam training text files, or ham
    /// training text files.
    /// ```no_run
    /// # use rammer::BagOfWords;
    /// let spam_bow = BagOfWords::from_folder("data/train/spam");
    /// ```
    pub fn from_folder(dir_path: &str) -> Option<Self> {
        let bow: BagOfWords = fs::read_dir(dir_path)
            .ok()?
            .par_bridge()
            .filter_map(|entry| {
                entry
                    .ok()
                    .and_then(|e| e.path().to_str().and_then(|p| BagOfWords::from_file(p)))
            })
            .collect();

        Some(bow)
    }

    /// Combines two BagOfWords into a new BagOfWords.
    /// Freqencies of words found in both bags are additive.
    /// This operation is commutative and associative. These properties can be used to dynamically
    /// grow your training BagOfWords.
    /// ```
    /// # use rammer::BagOfWords;
    /// let ham_bow_1 = BagOfWords::from("Hello there world"); // Creates: {HELLO: 1, THERE: 1, WORLD: 1}
    /// let ham_bow_2 = BagOfWords::from("howdy there guy"); // Creates: {HOWDY: 1, THERE: 1, GUY: 1}
    /// let com_bow = ham_bow_1.combine(ham_bow_2); // Combines to: {HELLO: 1, THERE: 2, HOWDY: 1, ...}
    /// ```
    pub fn combine(mut self, other: Self) -> Self {
        for (k, v) in other.bow {
            self.bow.entry(k).and_modify(|sv| *sv += v).or_insert(v);
        }
        self
    }

    /// Get the sum of all the Counts in a BagOfWords.
    /// Used internally for frequency calculations.
    /// ```
    /// # use rammer::BagOfWords;
    /// # let ham_bow = BagOfWords::new();
    /// ham_bow.total_word_count(); // returns a sum of Counts.
    /// ```
    pub fn total_word_count(&self) -> Count {
        self.bow.values().sum()
    }

    /// Calculates the Frequency of a word in the BagOfWords by taking count_of_a_word / total_word_count.
    /// This will return None, if the word slice passed contains multiple words.
    /// ```
    /// # use rammer::BagOfWords;
    /// let ham_bow = BagOfWords::from("hello there how are you");
    /// ham_bow.word_frequency("hello"); //returns 0.2
    /// ham_bow.word_frequency("hello there"); //returns None
    /// ```
    pub fn word_frequency(&self, word: &str) -> Option<Frequency> {
        let word_vec: Vec<&str> = word
            .split_word_bounds()
            .filter(|&s| !s.trim().is_empty())
            .collect();
        if word_vec.len() == 0 || word_vec.len() > 1 {
            return None;
        }

        self.bow
            .get(&word_vec[0].to_uppercase()[..])
            .and_then(|&v| Some(v as Frequency / self.total_word_count() as Frequency))
    }
}

/// Converts a &str to a bag of words.
/// This to create BagOfWord models, consider using [from_file](struct.BagOfWords.html#method.from_file) or
/// [from_folder](struct.BagOfWords.html#method.from_folder) instead.
/// ```
/// # use rammer::BagOfWords;
/// let bow = BagOfWords::from("hello world WOrLD"); // creates {HELLO: 1, WORLD: 2}
/// ```
impl convert::From<&str> for BagOfWords {
    #[allow(missing_doc_code_examples)]
    fn from(s: &str) -> BagOfWords {
        let mut bow = BagOfWords::new();
        for w in s.split_word_bounds().filter(|&s| !s.trim().is_empty()) {
            *bow.bow.entry(w.to_uppercase()).or_insert(0) += 1;
        }
        bow
    }
}

/// Use [.collect()](https://doc.rust-lang.org/std/iter/trait.Iterator.html#method.collect)
/// over an iterator of BagOfWords to additively combine them with [combine](struct.BagOfWords.html#method.combine)
/// ```
/// # use rammer::BagOfWords;
/// let bow: BagOfWords = vec![
///     BagOfWords::from("hi"),
///     BagOfWords::new(),
///     BagOfWords::from("Big sale!")]
///     .into_iter().collect();
/// ```
impl iter::FromIterator<BagOfWords> for BagOfWords {
    #[allow(missing_doc_code_examples)]
    fn from_iter<I: IntoIterator<Item = BagOfWords>>(iter: I) -> Self {
        let mut c = BagOfWords::new();
        for i in iter {
            c = c.combine(i);
        }
        c
    }
}

/// Use [.collect()](https://doc.rust-lang.org/std/iter/trait.Iterator.html#method.collect)
/// over a parallel iterator of BagOfWords to additively combine them with [combine](struct.BagOfWords.html#method.combine)
/// use [rayon](https://docs.rs/rayon/1.0.1/rayon/index.html) crate to make .into_par_iter()
/// available.
/// ```
/// # use rammer::BagOfWords;
/// use rayon::prelude::*;
/// let bow: BagOfWords = vec![
///     BagOfWords::from("hi"),
///     BagOfWords::new(),
///     BagOfWords::from("Big sale!")]
///     .into_par_iter().collect();
/// ```
impl FromParallelIterator<BagOfWords> for BagOfWords {
    #[allow(missing_doc_code_examples)]
    fn from_par_iter<I>(par_iter: I) -> Self
    where
        I: IntoParallelIterator<Item = BagOfWords>,
    {
        //let par_iter = par_iter.into_par_iter();
        par_iter
            .into_par_iter()
            .reduce(|| BagOfWords::new(), |a, b| a.combine(b))
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
        let fbow: BagOfWords =
            BagOfWords::from_file("test_resources/test_data/ascii_only.txt").unwrap();
        let bow = BagOfWords::from("HELLO THERE WORLD");
        assert_eq!(fbow, bow);
    }

    #[test]
    fn bow_from_file_unicode_only() {
        let fbow: BagOfWords =
            BagOfWords::from_file("test_resources/test_data/unicode_only.txt").unwrap();
        let bow = BagOfWords::from("😊😊😊😊😊");
        assert_eq!(fbow, bow);
    }

    #[test]
    fn bow_from_file_unicode_and_ascii() {
        let fbow: BagOfWords =
            BagOfWords::from_file("test_resources/test_data/unicode_and_ascii.txt").unwrap();
        let bow = BagOfWords::from("😊😊😊😊😊 HELLO THERE WORLD");
        assert_eq!(fbow, bow);
    }

    /*****************************************/
    /* FROM FOLDER TESTS                     */
    /*****************************************/

    #[test]
    fn bow_from_test_data_folder() {
        let fbow: BagOfWords =
            BagOfWords::from_folder("test_resources/test_data").expect("Folder not found");
        let bow = BagOfWords::new()
            .combine(BagOfWords::from("hello there world"))
            .combine(BagOfWords::from("hello there world 😊😊😊😊😊"))
            .combine(BagOfWords::from("😊😊😊😊😊"));

        assert_eq!(fbow, bow);
    }

    /*****************************************/
    /* WORD_FREQUENCY TESTS                  */
    /*****************************************/
    #[test]
    fn freq_1() {
        let bow = BagOfWords::from("hello hello hello hello");
        assert_eq!(bow.word_frequency("hello").unwrap(), 1.0f64);
    }

    #[test]
    fn freq_0() {
        let bow = BagOfWords::from("hello hello hello hello");
        assert!(bow.word_frequency("there").is_none());
    }

    #[test]
    fn freq_1_of_2() {
        let bow = BagOfWords::from("hello there");
        assert_eq!(bow.word_frequency("hello").unwrap(), 0.5f64);
    }

    #[test]
    fn freq_1_of_5() {
        let bow = BagOfWords::from("hello there you cutie pie");
        assert_eq!(bow.word_frequency("hello").unwrap(), 0.2f64);
    }
}
