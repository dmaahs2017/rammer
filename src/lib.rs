#![warn(missing_docs, missing_doc_code_examples)]
//! Rammer, a play on Rust and the fact that spam is classified as Spam or Ham, is a spam/ham
//! classification library.
//! ```
//! use rammer::{HSModel, BagOfWords};
//! let spam_bow = BagOfWords::from_folder("data/train/spam");
//! let ham_bow = BagOfWords::from_folder("data/train/ham");
//! let model = HSModel::from_bows(ham_bow, spam_bow);
//! model.text_spam_probability("hello i have an offer for you");
//! model.text_spam_probability("Hey it's greg, finished the data analysis");
//! ```  

mod bag_of_words;
mod hs_model;
pub use bag_of_words::BagOfWords;
pub use hs_model::HSModel;

/// Type alias for rate of occurences of a value.
/// This type should always be between [0,1].
pub type Frequency = f64;

/// Type alias for the statistical probability of an event.
/// This type should always be between [0,1].
pub type Probability = f64;

/// Type alias for number of times a word is found in a BagOfWords.
pub type Count = u32;
