use rammer::{HSModel, BagOfWords};

fn main() {
    let spam_bow = BagOfWords::read_from_csv("filtered_spam.csv").unwrap();
    let ham_bow = BagOfWords::read_from_csv("filtered_ham.csv").unwrap();
    let model = HSModel {
        spam_bow,
        ham_bow,
    };

    // Use the Model
    let p = model.text_spam_probability("free money");
    println!("Spam Chance: {}", p)
}
