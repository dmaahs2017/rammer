use std::collections::HashMap;
use std::convert;
use unicode_segmentation::UnicodeSegmentation;

#[derive(PartialEq, Eq, Debug)]
pub struct BagOfWords {
    bow: HashMap<String, u32>,
}

impl BagOfWords {
    pub fn new() -> Self {
        BagOfWords {
            bow: HashMap::new(),
        }
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


#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /*****************************************/
    /* FROM &str TESTING */
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
}
