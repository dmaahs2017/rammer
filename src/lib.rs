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

mod bayes;
pub use bayes::{HSModel, BagOfWords};
