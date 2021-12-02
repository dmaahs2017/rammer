use rammer::HSModel;
use rayon::prelude::*;
use std::fs;

fn main() {
    let model = HSModel::read_from_json("out/models/enron1_model.json").unwrap();

    let top_ten = model.top_ten_disproportionately_ham();

    for (word , probability) in &top_ten {
        println!("'{}'", word);
        println!("Spam Chance: {}", probability);
        println!("Spam Freq: {}", model.spam_bow.word_frequency(&word).unwrap());
        println!("Ham Freq: {}", model.ham_bow.word_frequency(&word).unwrap());
        println!();
    }

    //println!("Spam Chance: {}", model.text_spam_probability("credit"));
    //println!(
    //    "Spam Freq: {}",
    //    model.spam_bow.word_frequency("credit").unwrap()
    //);
    //println!(
    //    "Ham: Freq: {}",
    //    model.ham_bow.word_frequency("credit").unwrap()
    //);

    //let top_ten = model.top_10_ham_freq();
    //dbg!(top_ten);

    //let top_ten_spam = model.spam_bow.top_10_count();
    //dbg!(top_ten_spam);

    //println!(
    //    "Spam Correctly Classified: {}/{} = {:.4}",
    //    spam_answers.0, spam_answers.1, spam_answers.2
    //);
    //println!(
    //    "Ham Correctly Classified: {}/{} = {:.4}",
    //    ham_answers.0, ham_answers.1, ham_answers.2
    //);
}

fn validate<F>(model: &HSModel, dir: &str, class: &str, is_correct: F) -> (u32, usize, f64)
where
    F: Fn(f64) -> bool + Sync,
{
    let ps: Vec<bool> = fs::read_dir(dir)
        .expect("folder exists")
        .par_bridge()
        .filter_map(|maybe_entry| {
            maybe_entry.ok().and_then(|entry| {
                fs::read_to_string(entry.path())
                    .ok()
                    .and_then(|text| Some(model.text_spam_probability(&text[..])))
            })
        })
        .map(|p| {
            println!("Probability: {:.8}\t\t({})", p, class);
            is_correct(p)
        })
        .collect();

    let num_classified_correctly: u32 = ps
        .iter()
        .filter_map(|&b| if b { Some(1) } else { None })
        .sum();

    (
        num_classified_correctly,
        ps.len(),
        num_classified_correctly as f64 / ps.len() as f64,
    )
}
