use rammer::BagOfWords;
use std::io::prelude::*;
use std::fs::File;

fn main() {
    let spam_bow = BagOfWords::from_folder("data/train/spam");
    let serialized = serde_json::to_string(&spam_bow).unwrap();
    let mut file = File::create("out/spam.json").expect("file to open");
    file.write_all(&serialized.into_bytes()[..]).expect("Serialized");
}
