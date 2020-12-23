#![warn(missing_docs, missing_doc_code_examples)]
//! Rammer, a play on Rust and the fact that spam is classified as Spam or Ham, is a spam/ham
//! classification library.
//!
//! Here is an example program which trains and saves a new model for later use.
//! ```no_run
//! use rammer::{ HSModel, BagOfWords };
//!
//! fn main() {
//!     let spam_bow = BagOfWords::from_folder("data/train/spam");
//!     let ham_bow = BagOfWords::from_folder("data/train/ham");
//!     let model = HSModel::from_bows(ham_bow, spam_bow);
//!     model.write_to_json("out/models/enron1_model.json");
//! }
//! ```
//!
//! Here is an Example program using an existing model.
//! ```no_run
//! use rammer::HSModel;
//! use std::fs;
//! use rayon::prelude::*;
//! fn main() {
//!    let model = HSModel::read_from_json("out/models/enron1_model.json").unwrap();
//!    let spam_answers = validate(&model, "data/validate/spam", "spam", |p| p > 0.8);
//!    let ham_answers = validate(&model, "data/validate/ham", "ham", |p| p < 0.2);
//!
//!    println!("Spam Correctly Classified: {}/{} = {:.4}", spam_answers.0, spam_answers.1, spam_answers.2);
//!    println!("Ham Correctly Classified: {}/{} = {:.4}", ham_answers.0, ham_answers.1, ham_answers.2);
//! }
//!
//! fn validate<F>(model: &HSModel, dir: &str, class: &str, is_correct: F) -> (u32, usize, f64)
//!     where F: Fn(f64) -> bool + Sync
//! {
//!     let ps: Vec<bool> = fs::read_dir(dir)
//!         .expect("folder exists")
//!         .par_bridge()
//!         .filter_map(|maybe_entry| {
//!             maybe_entry.ok().and_then(|entry| {
//!                 fs::read_to_string(entry.path())
//!                     .ok()
//!                     .and_then(|text| Some(model.text_spam_probability(&text[..])))
//!             })
//!         })
//!         .map(|p| { println!("Probability: {:.8}\t\t({})", p, class); is_correct(p) })
//!         .collect();
//!
//!     let num_classified_correctly: u32 = ps
//!         .iter()
//!         .filter_map(|&b| if b { Some(1) } else { None })
//!         .sum();
//!
//!     (
//!         num_classified_correctly,
//!         ps.len(),
//!         num_classified_correctly as f64 / ps.len() as f64
//!     )
//!
//! }
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
