mod learning;
mod math;
mod neural;
mod experiments;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    match args.first().map(String::as_str) {
        Some("lineal") => experiments::linear::run(),
        Some("booleano") => experiments::true_false::run(),
        Some("vocal") => experiments::vowels::run(args.get(1).map(String::as_str)),
        Some(value) => experiments::vowels::run(Some(value)),
        None => experiments::vowels::run(None),
    }
}
