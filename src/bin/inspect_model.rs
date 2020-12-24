use rammer::HSModel;

fn main() {
    let model = HSModel::read_from_json("out/models/enron1_model.json").unwrap();
    println!("{}", serde_json::to_string_pretty(&model).unwrap());
}
