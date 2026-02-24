pub mod algebraic;
pub mod benchmarks;
pub mod checker;
pub mod fence;
pub mod hardware;
pub mod integration;
pub mod llm;
pub mod models;
pub mod security;
pub mod symmetry;
pub mod frontend;
pub mod utils;
pub mod testgen;

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
