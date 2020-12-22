use unicode_segmentation::UnicodeSegmentation;
use crate::{ BagOfWords, Probability, Frequency };

/// A model which contains 2 BagOfWords, one containing known spam, and the other known ham.
/// ```
/// # use rammer::{BagOfWords, HSModel};
/// let ham_bow = BagOfWords::from("hello there how are you");
/// let spam_bow = BagOfWords::from("I have an offer you won't be able to pass up!!!");
/// let model = HSModel::new().add_spam_bow(spam_bow).add_ham_bow(ham_bow);
/// ```
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
}

#[cfg(test)]
mod tests {
    use crate::BagOfWords;
    use super::HSModel;

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
