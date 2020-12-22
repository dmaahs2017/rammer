use rammer::{ HSModel, BagOfWords };

fn main() {
    let spam_bow = BagOfWords::from_folder("data/train/spam");
    let ham_bow = BagOfWords::from_folder("data/train/ham");
    let model = HSModel::from_bows(ham_bow, spam_bow);
    dbg!(model.text_spam_probability("hello i have an offer for you"));
    dbg!(model.text_spam_probability("Hey it's greg, finished the data analysis"));
}
