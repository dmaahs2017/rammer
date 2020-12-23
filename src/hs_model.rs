use std::fs;

use serde::{Deserialize, Serialize};
use unicode_segmentation::UnicodeSegmentation;

use crate::{BagOfWords, Frequency, Probability};

/// A model which contains 2 BagOfWords, one containing known spam, and the other known ham.
/// ```
/// # use rammer::{BagOfWords, HSModel};
/// let ham_bow = BagOfWords::from("hello there how are you");
/// let spam_bow = BagOfWords::from("I have an offer you won't be able to pass up!!!");
/// let model = HSModel::new().add_spam_bow(spam_bow).add_ham_bow(ham_bow);
/// ```
#[derive(Serialize, Deserialize)]
pub struct HSModel {
    ham_bow: BagOfWords,
    spam_bow: BagOfWords,
}

#[allow(missing_doc_code_examples)]
impl HSModel {
    /// Create a new empty model, with no training data.
    /// ```
    /// # use rammer::{BagOfWords, HSModel};
    /// let model = HSModel::new(); //returns an empty model.
    /// ```
    pub fn new() -> Self {
        HSModel {
            ham_bow: BagOfWords::new(),
            spam_bow: BagOfWords::new(),
        }
    }

    /// Builder pattern for adding a spam_bow with the [combine](struct.BagOfWords.html#method.combine) method.
    /// ```
    /// # use rammer::{BagOfWords, HSModel};
    /// # let spam_bow = BagOfWords::from("I have an offer you won't be able to pass up!!!");
    /// # let ham_bow = BagOfWords::from("How are you today.");
    /// let model = HSModel::new().add_spam_bow(spam_bow).add_ham_bow(ham_bow); //builder pattern
    /// ```
    pub fn add_spam_bow(mut self, spam_bow: BagOfWords) -> Self {
        self.spam_bow = self.spam_bow.combine(spam_bow);
        self
    }

    /// Builder pattern for adding a ham_bow with the [combine](struct.BagOfWords.html#method.combine) method.
    /// ```
    /// # use rammer::{BagOfWords, HSModel};
    /// # let ham_bow = BagOfWords::from("How are you today.");
    /// # let spam_bow = BagOfWords::from("I have an offer you won't be able to pass up!!!");
    /// let model = HSModel::new().add_ham_bow(ham_bow).add_spam_bow(spam_bow); //builder pattern
    /// ```
    pub fn add_ham_bow(mut self, ham_bow: BagOfWords) -> Self {
        self.ham_bow = self.ham_bow.combine(ham_bow);
        self
    }

    /// Create a [HSModel](HSModel) from a ham_bow and a spam_bow.
    /// ```
    /// # use rammer::{BagOfWords, HSModel};
    /// # let ham_bow = BagOfWords::from("How are you today.");
    /// # let spam_bow = BagOfWords::from("I have an offer you won't be able to pass up!!!");
    /// let model = HSModel::from_bows(ham_bow, spam_bow);
    /// ```
    pub fn from_bows(ham_bow: BagOfWords, spam_bow: BagOfWords) -> Self {
        HSModel::new().add_ham_bow(ham_bow).add_spam_bow(spam_bow)
    }

    /// Returns the probability that a slice of text is spam, based on the model.
    /// Read about how this is calulated here on the
    /// [Naive Bayes Spam Filtering Wikipedia Page](https://en.wikipedia.org/wiki/Naive_Bayes_spam_filtering)
    /// ```
    /// # use rammer::{BagOfWords, HSModel};
    /// # let ham_bow = BagOfWords::from("How are you today.");
    /// # let spam_bow = BagOfWords::from("I have an offer you won't be able to pass up!!!");
    /// # let model = HSModel::from_bows(ham_bow, spam_bow);
    /// let spam_probability = model.text_spam_probability("Respond fast! I have an offer of a lifetime!"); // return value between [0.0, 1.0]
    /// ```
    pub fn text_spam_probability(&self, text: &str) -> Probability {
        let n: f64 = text
            .to_uppercase()
            .split_word_bounds()
            .filter(|&s| !s.trim().is_empty())
            .filter_map(|word| {
                if let (Some(spam_freq), Some(ham_freq)) = (
                    self.spam_bow.word_frequency(word),
                    self.ham_bow.word_frequency(word),
                ) {
                    let p = spam_freq / (spam_freq + ham_freq);
                    Some(Frequency::ln(1.0 - p) - Frequency::ln(p))
                } else {
                    None
                }
            })
            .sum();
        1.0 / (1.0 + std::f64::consts::E.powf(n))
    }

    /// Serializse HSModel to a compact json string and write it to file_path. This write is
    /// destructive.
    /// ```
    /// # use rammer::{HSModel, BagOfWords};
    /// # let model = HSModel::from_bows(BagOfWords::from("hi greetings afternoon well"), BagOfWords::from("buy pay sell free"));
    /// model.write_to_json("test_resources/test_models/model.json");
    /// ```
    pub fn write_to_json(&self, file_path: &str) -> () {
        if let Ok(serialized) = serde_json::to_string(self) {
            fs::write(file_path, serialized).unwrap();
        }
    }

    /// read a json string from file_path and deserialize it to HSModel.
    /// ```
    /// # use rammer::{HSModel, BagOfWords};
    /// let model = HSModel::read_from_json("test_resources/test_models/model.json").unwrap();
    /// ```
    pub fn read_from_json(file_path: &str) -> Option<Self> {
        if let Ok(serialized) = fs::read_to_string(file_path) {
            serde_json::from_str(&serialized[..]).ok()
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::HSModel;
    use crate::BagOfWords;

    /*****************************************/
    /* HSModel TESTS                         */
    /*****************************************/

    #[test]
    fn filter_test() {
        let spam_bow = BagOfWords::from("spam spam spam spam ham");
        let ham_bow = BagOfWords::from("spam ham");
        let model = HSModel::from_bows(ham_bow, spam_bow);
        assert!(model.text_spam_probability("spam") >= 0.0);
        assert!(model.text_spam_probability("spam") <= 1.0);
    }
}
