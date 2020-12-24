use rammer::{BagOfWords, HSModel};

fn main() {
    let spam_bow = BagOfWords::from_folder("data/train/spam");
    let ham_bow = BagOfWords::from_folder("data/train/ham");
    let model = HSModel::from_bows(ham_bow, spam_bow);
    model.write_to_json("out/models/enron1_model.json");
}
