use rammer::{ HSModel, BagOfWords };
use std::io::prelude::*;
use std::fs::File;

fn main() {
    let spam_bow = BagOfWords::from_folder("data/train/spam");
    let ham_bow = BagOfWords::from_folder("data/train/ham");
    let model = HSModel::from_bows(ham_bow, spam_bow);
    dbg!(model.text_spam_probability("angels scrub lmao goteem"));
}
