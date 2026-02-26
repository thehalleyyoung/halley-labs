//! Comprehensive Benchmark Runner
//!
//! Main experiment for the Spectacles paper. Runs ALL metrics through the
//! triple verification pipeline on a diverse corpus of 200+ input pairs,
//! then certifies representative inputs via the STARK pipeline.
//!
//! Outputs:
//!   - Detailed table to stderr
//!   - Full JSON to stdout
//!   - `comprehensive_benchmark_results.json` at the implementation root

use spectacles_core::scoring::{
    ScoringPair, TripleMetric,
    exact_match::ExactMatchScorer,
    token_f1::TokenF1Scorer,
    bleu::{BleuScorer, SmoothingMethod},
    rouge::{RougeNScorer, RougeLScorer},
    chrf::{ChrFScorer, ChrFConfig},
    differential::{DifferentialTester, standard_test_suite, random_test_pairs},
};
use spectacles_core::pipeline::certify_metric;
use serde::Serialize;
use std::time::Instant;

// ═══════════════════════════════════════════════════════════════════════════
// Result types
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Serialize)]
struct ComprehensiveResults {
    timestamp: String,
    summary: Summary,
    corpus_description: CorpusDescription,
    metric_results: Vec<MetricBenchmark>,
    per_category_results: Vec<CategoryResults>,
    certification_results: Vec<CertificationResult>,
    differential_test_results: DifferentialTestSummary,
    timing: TimingBreakdown,
}

#[derive(Debug, Serialize)]
struct Summary {
    total_input_pairs: usize,
    total_metrics: usize,
    total_triple_checks: usize,
    total_disagreements: usize,
    overall_agreement_rate: f64,
    all_metrics_agree: bool,
    total_certifications: usize,
    certifications_verified: usize,
    total_wall_clock_ms: f64,
}

#[derive(Debug, Serialize)]
struct CorpusDescription {
    categories: Vec<CategoryInfo>,
    total_pairs: usize,
}

#[derive(Debug, Serialize)]
struct CategoryInfo {
    name: String,
    count: usize,
    description: String,
}

#[derive(Debug, Serialize)]
struct MetricBenchmark {
    metric_name: String,
    total_pairs: usize,
    mean_score: f64,
    min_score: f64,
    max_score: f64,
    std_dev: f64,
    triple_agreements: usize,
    triple_disagreements: usize,
    agreement_rate: f64,
    total_time_ms: f64,
    per_pair_time_us: f64,
}

#[derive(Debug, Serialize)]
struct CategoryResults {
    category: String,
    num_pairs: usize,
    metrics: Vec<CategoryMetricResult>,
}

#[derive(Debug, Serialize)]
struct CategoryMetricResult {
    metric: String,
    mean_score: f64,
    agreement_rate: f64,
    time_ms: f64,
}

#[derive(Debug, Serialize)]
struct CertificationResult {
    metric: String,
    candidate_preview: String,
    reference_preview: String,
    score: f64,
    proof_generated: bool,
    proof_verified: bool,
    triple_agreement: bool,
    prove_time_ms: f64,
    verify_time_ms: f64,
    proof_size_bytes: usize,
    num_wfa_states: usize,
    num_constraints: usize,
    trace_length: usize,
}

#[derive(Debug, Serialize)]
struct DifferentialTestSummary {
    num_standard_pairs: usize,
    num_random_pairs: usize,
    per_metric: Vec<DiffMetricSummary>,
    total_checks: usize,
    total_disagreements: usize,
}

#[derive(Debug, Serialize)]
struct DiffMetricSummary {
    metric: String,
    total_tests: usize,
    agreements: usize,
    disagreements: usize,
    agreement_rate: f64,
}

#[derive(Debug, Serialize)]
struct TimingBreakdown {
    total_ms: f64,
    corpus_generation_ms: f64,
    triple_scoring_ms: f64,
    certification_ms: f64,
    differential_testing_ms: f64,
    per_metric_ms: Vec<(String, f64)>,
}

// ═══════════════════════════════════════════════════════════════════════════
// Corpus generation (200+ diverse pairs)
// ═══════════════════════════════════════════════════════════════════════════

struct TaggedPair {
    pair: ScoringPair,
    category: &'static str,
}

fn build_corpus() -> Vec<TaggedPair> {
    let mut corpus = Vec::new();

    // --- MMLU-style Q&A (50 pairs) ---
    let mmlu_data: Vec<(&str, &str)> = vec![
        ("A", "A"), ("B", "A"), ("C", "C"), ("D", "B"), ("A", "A"),
        ("B", "C"), ("C", "C"), ("A", "D"), ("D", "D"), ("B", "B"),
        ("A", "B"), ("C", "A"), ("D", "D"), ("A", "A"), ("B", "B"),
        ("C", "D"), ("A", "A"), ("D", "C"), ("B", "B"), ("A", "A"),
        ("the mitochondria", "the mitochondria"),
        ("photosynthesis", "photosynthesis"),
        ("osmosis", "diffusion"),
        ("42", "42"),
        ("1776", "1789"),
        ("Newton", "Newton"),
        ("Einstein", "Bohr"),
        ("helium", "helium"),
        ("iron", "copper"),
        ("Paris", "Paris"),
        ("gravity", "gravity"),
        ("evolution", "evolution"),
        ("plate tectonics", "plate tectonics"),
        ("DNA replication", "DNA replication"),
        ("quantum mechanics", "quantum physics"),
        ("thermodynamics", "thermodynamics"),
        ("relativity", "general relativity"),
        ("natural selection", "natural selection"),
        ("covalent bond", "ionic bond"),
        ("mitosis", "meiosis"),
        ("proton", "proton"),
        ("electron", "neutron"),
        ("velocity", "speed"),
        ("acceleration", "acceleration"),
        ("momentum", "impulse"),
        ("wavelength", "frequency"),
        ("amplitude", "amplitude"),
        ("resistance", "impedance"),
        ("capacitance", "capacitance"),
        ("inductance", "inductance"),
    ];
    for (c, r) in &mmlu_data {
        corpus.push(TaggedPair {
            pair: ScoringPair { candidate: c.to_string(), reference: r.to_string() },
            category: "mmlu_qa",
        });
    }

    // --- SQuAD-style extractive QA (50 pairs) ---
    let squad_data: Vec<(&str, &str)> = vec![
        ("the quick brown fox", "the quick brown fox jumps over the lazy dog"),
        ("machine learning is a subset of artificial intelligence", "machine learning is a branch of artificial intelligence"),
        ("the capital of France is Paris", "Paris is the capital of France"),
        ("water boils at 100 degrees Celsius", "water boils at 100 degrees Celsius at standard pressure"),
        ("DNA stands for deoxyribonucleic acid", "DNA stands for deoxyribonucleic acid"),
        ("the speed of light is approximately 300000 km per second", "the speed of light in vacuum is approximately 299792 kilometers per second"),
        ("Shakespeare wrote Hamlet", "William Shakespeare authored the play Hamlet"),
        ("the Earth orbits the Sun", "the Earth revolves around the Sun"),
        ("gravity is a fundamental force", "gravity is one of the four fundamental forces of nature"),
        ("hydrogen is the lightest element", "hydrogen is the lightest and most abundant element in the universe"),
        ("the human body has 206 bones", "an adult human body contains 206 bones"),
        ("pi is approximately 3.14159", "the value of pi is approximately 3.14159265"),
        ("photosynthesis converts light energy into chemical energy", "photosynthesis is the process by which plants convert light energy into chemical energy"),
        ("the Pythagorean theorem relates sides of a right triangle", "in a right triangle the square of the hypotenuse equals the sum of the squares of the other two sides"),
        ("mitosis is cell division", "mitosis is a type of cell division that results in two identical daughter cells"),
        ("the Great Wall of China is a historical structure", "the Great Wall of China is a series of fortifications along the northern borders of China"),
        ("oxygen is essential for respiration", "oxygen is required for cellular respiration in aerobic organisms"),
        ("the periodic table organizes elements", "the periodic table arranges chemical elements by atomic number"),
        ("neurons transmit electrical signals", "neurons are cells that transmit electrical and chemical signals in the nervous system"),
        ("evolution is driven by natural selection", "evolution occurs through the mechanism of natural selection acting on genetic variation"),
        ("the Amazon River is the longest river", "the Amazon River is the largest river by discharge volume"),
        ("insulin regulates blood sugar", "insulin is a hormone that regulates glucose levels in the blood"),
        ("tectonic plates move slowly", "tectonic plates drift at rates of a few centimeters per year"),
        ("antibiotics treat bacterial infections", "antibiotics are medications used to treat infections caused by bacteria"),
        ("the Sun is a star", "the Sun is a medium sized star in the Milky Way galaxy"),
        ("volcanoes form at plate boundaries", "volcanic activity occurs primarily at convergent and divergent plate boundaries"),
        ("the Moon causes tides", "tidal forces are primarily caused by the gravitational pull of the Moon"),
        ("chlorophyll absorbs light", "chlorophyll is the pigment that absorbs light energy during photosynthesis"),
        ("sound travels faster in water", "sound waves propagate faster through water than through air"),
        ("the brain controls body functions", "the brain is the central organ of the nervous system controlling bodily functions"),
        ("diamonds are made of carbon", "diamonds consist of carbon atoms arranged in a crystal lattice"),
        ("RNA carries genetic information", "RNA serves as a messenger carrying genetic instructions from DNA"),
        ("earthquakes occur along fault lines", "seismic activity is concentrated along geological fault lines"),
        ("the ozone layer protects from UV", "the ozone layer in the stratosphere absorbs ultraviolet radiation"),
        ("glaciers are made of compressed snow", "glaciers form from the accumulation and compaction of snow over many years"),
        ("iron is magnetic", "iron is a ferromagnetic element that can be magnetized"),
        ("the equator divides Earth in half", "the equator is an imaginary line dividing Earth into northern and southern hemispheres"),
        ("fossils preserve ancient life", "fossils are the preserved remains or traces of ancient organisms"),
        ("the Sahara is the largest desert", "the Sahara Desert is the largest hot desert in the world"),
        ("black holes have strong gravity", "black holes are regions of spacetime with gravitational fields so strong that nothing can escape"),
        ("coral reefs support marine life", "coral reefs are underwater ecosystems that support a diverse array of marine species"),
        ("the atmosphere contains nitrogen and oxygen", "Earth atmosphere is composed primarily of nitrogen and oxygen"),
        ("comets orbit the Sun", "comets are icy bodies that orbit the Sun in elliptical paths"),
        ("enzymes catalyze reactions", "enzymes are biological catalysts that speed up chemical reactions"),
        ("the Mariana Trench is the deepest point", "the Mariana Trench is the deepest known location in Earth oceans"),
        ("vaccines prevent diseases", "vaccines stimulate the immune system to provide protection against specific diseases"),
        ("algae produce oxygen", "algae are photosynthetic organisms that produce a significant portion of Earth oxygen"),
        ("continental drift shapes geography", "continental drift is the gradual movement of continents across Earth surface over geological time"),
        ("DNA has a double helix structure", "DNA is structured as a double helix of nucleotide base pairs"),
        ("metamorphic rocks form under pressure", "metamorphic rocks are formed when existing rocks are transformed by heat and pressure"),
    ];
    for (c, r) in &squad_data {
        corpus.push(TaggedPair {
            pair: ScoringPair { candidate: c.to_string(), reference: r.to_string() },
            category: "squad_qa",
        });
    }

    // --- Translation-style pairs (40 pairs) ---
    let translation_data: Vec<(&str, &str)> = vec![
        ("The cat is on the mat", "The cat is sitting on the mat"),
        ("I like to eat apples and bananas", "I enjoy eating apples and bananas"),
        ("The weather is nice today", "Today the weather is beautiful"),
        ("She went to the store to buy milk", "She went to the shop to purchase milk"),
        ("The book is very interesting", "The book is extremely engaging"),
        ("He runs every morning in the park", "Every morning he goes running in the park"),
        ("The students are studying for their exams", "The students are preparing for their examinations"),
        ("We need to find a solution to this problem", "We must discover a solution for this issue"),
        ("The restaurant serves delicious food", "The restaurant offers wonderful cuisine"),
        ("Technology is changing the world rapidly", "Technology is transforming the world at a rapid pace"),
        ("The children are playing in the garden", "The kids are playing outside in the garden"),
        ("Learning a new language takes time and effort", "Acquiring a new language requires time and dedication"),
        ("The movie received positive reviews from critics", "The film was well received by critics"),
        ("Scientists discovered a new species of fish", "Researchers found a previously unknown species of fish"),
        ("The meeting has been postponed until next week", "The meeting was rescheduled to the following week"),
        ("The train arrives at noon", "The train is scheduled to arrive at twelve o clock"),
        ("She speaks three languages fluently", "She is fluent in three different languages"),
        ("The river flows through the valley", "The river runs through the middle of the valley"),
        ("He finished the project on time", "He completed the project before the deadline"),
        ("The flowers bloom in spring", "Flowers blossom during the spring season"),
        ("The museum has many ancient artifacts", "The museum contains numerous historical artifacts"),
        ("She plays the piano beautifully", "She performs piano music with great skill"),
        ("The company launched a new product", "The firm introduced a new product to the market"),
        ("He reads a book every week", "He finishes reading one book per week"),
        ("The city has a large population", "The city is home to a significant number of residents"),
        ("She won first place in the competition", "She took the top prize at the contest"),
        ("The garden needs watering", "The garden requires regular irrigation"),
        ("He drove to work this morning", "He commuted to the office by car today"),
        ("The airplane landed safely", "The aircraft touched down without incident"),
        ("She wrote a letter to her friend", "She composed a letter addressed to her friend"),
        ("The bridge connects two cities", "The bridge links the two neighboring cities"),
        ("He solved the puzzle quickly", "He figured out the puzzle in a short time"),
        ("The dog chased the cat", "The dog ran after the cat"),
        ("She painted a beautiful landscape", "She created a stunning landscape painting"),
        ("The library has thousands of books", "The library holds a vast collection of books"),
        ("He fixed the broken window", "He repaired the shattered window"),
        ("The concert was amazing", "The musical performance was outstanding"),
        ("She baked a chocolate cake", "She made a cake with chocolate"),
        ("The road leads to the mountains", "The road goes toward the mountain range"),
        ("He taught mathematics at the university", "He was a mathematics professor at the university"),
    ];
    for (c, r) in &translation_data {
        corpus.push(TaggedPair {
            pair: ScoringPair { candidate: c.to_string(), reference: r.to_string() },
            category: "translation",
        });
    }

    // --- Code completion / HumanEval-style (40 pairs) ---
    let code_data: Vec<(&str, &str)> = vec![
        ("def add(a, b): return a + b", "def add(a, b): return a + b"),
        ("def multiply(x, y): return x * y", "def multiply(a, b): return a * b"),
        ("for i in range(10): print(i)", "for i in range(10): print(i)"),
        ("if x > 0: return True", "if x > 0: return True else: return False"),
        ("import numpy as np", "import numpy as np"),
        ("class Node: def __init__(self, val): self.val = val", "class TreeNode: def __init__(self, value): self.value = value"),
        ("def factorial(n): return 1 if n <= 1 else n * factorial(n-1)", "def factorial(n): return 1 if n == 0 else n * factorial(n - 1)"),
        ("sorted(arr, key=lambda x: x[1])", "sorted(arr, key=lambda item: item[1])"),
        ("with open('file.txt', 'r') as f: data = f.read()", "with open('file.txt') as f: content = f.read()"),
        ("result = [x**2 for x in range(10)]", "squares = [x * x for x in range(10)]"),
        ("def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)", "def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)"),
        ("try: x = int(input()) except ValueError: x = 0", "try: x = int(input()) except: x = 0"),
        ("dict_comp = {k: v for k, v in items}", "dictionary = {key: val for key, val in items}"),
        ("print('hello world')", "print('hello world')"),
        ("return sum(lst) / len(lst)", "return sum(numbers) / len(numbers)"),
        ("def is_prime(n): return all(n % i != 0 for i in range(2, n))", "def is_prime(n): return n > 1 and all(n % i for i in range(2, int(n**0.5)+1))"),
        ("arr.append(item)", "arr.append(item)"),
        ("os.path.join(dir, file)", "os.path.join(directory, filename)"),
        ("json.dumps(data, indent=2)", "json.dumps(data, indent=2)"),
        ("assert len(result) == expected", "assert len(result) == expected_length"),
        ("fn main() { println!(\"hello\"); }", "fn main() { println!(\"hello\"); }"),
        ("let x: i32 = 42;", "let x: i32 = 42;"),
        ("Vec::new()", "Vec::new()"),
        ("impl Display for Point { fn fmt(&self, f: &mut Formatter) -> Result { write!(f, \"({}, {})\", self.x, self.y) } }", "impl std::fmt::Display for Point { fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { write!(f, \"({}, {})\", self.x, self.y) } }"),
        ("match opt { Some(v) => v, None => default }", "match opt { Some(val) => val, None => default_value }"),
        ("use std::collections::HashMap;", "use std::collections::HashMap;"),
        ("struct Config { pub timeout: u64, pub retries: u32 }", "struct Config { timeout: u64, retries: u32 }"),
        ("async fn fetch(url: &str) -> Result<String> { Ok(reqwest::get(url).await?.text().await?) }", "async fn fetch(url: &str) -> anyhow::Result<String> { let resp = reqwest::get(url).await?; Ok(resp.text().await?) }"),
        ("let mut v = vec![3, 1, 4, 1, 5]; v.sort();", "let mut v = vec![3, 1, 4, 1, 5]; v.sort_unstable();"),
        ("fn gcd(a: u64, b: u64) -> u64 { if b == 0 { a } else { gcd(b, a % b) } }", "fn gcd(mut a: u64, mut b: u64) -> u64 { while b != 0 { let t = b; b = a % b; a = t; } a }"),
        ("SELECT * FROM users WHERE active = 1", "SELECT * FROM users WHERE active = true"),
        ("INSERT INTO logs (msg, ts) VALUES (?, NOW())", "INSERT INTO logs (message, timestamp) VALUES ($1, CURRENT_TIMESTAMP)"),
        ("CREATE TABLE items (id SERIAL PRIMARY KEY, name TEXT NOT NULL)", "CREATE TABLE items (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL)"),
        ("console.log('hello')", "console.log('hello')"),
        ("const x = arr.map(i => i * 2)", "const doubled = arr.map(item => item * 2)"),
        ("fetch('/api/data').then(r => r.json())", "fetch('/api/data').then(response => response.json())"),
        ("const [a, ...rest] = arr", "const [first, ...remaining] = arr"),
        ("Object.keys(obj).forEach(k => console.log(k))", "Object.keys(obj).forEach(key => console.log(key))"),
        ("async function getData() { return await fetch(url) }", "async function getData() { const res = await fetch(url); return res; }"),
        ("export default function App() { return <div>Hello</div> }", "export default function App() { return <div>Hello</div>; }"),
    ];
    for (c, r) in &code_data {
        corpus.push(TaggedPair {
            pair: ScoringPair { candidate: c.to_string(), reference: r.to_string() },
            category: "code_completion",
        });
    }

    // --- Edge cases (25 pairs) ---
    let long_fox = "the quick brown fox jumps over the lazy dog ".repeat(20);
    let long_alpha_a = "alpha beta gamma delta ".repeat(25);
    let long_alpha_b = "alpha beta gamma delta epsilon ".repeat(20);
    let long_word = "word ".repeat(100);

    let edge_pairs: Vec<(String, String)> = vec![
        // Empty strings
        ("".into(), "".into()),
        ("hello".into(), "".into()),
        ("".into(), "world".into()),
        // Single tokens
        ("a".into(), "a".into()),
        ("a".into(), "b".into()),
        ("x".into(), "x".into()),
        // Very long strings
        (long_fox.trim().to_string(), long_fox.trim().to_string()),
        (long_alpha_a.trim().to_string(), long_alpha_b.trim().to_string()),
        (long_word.trim().to_string(), long_word.trim().to_string()),
        // Unicode
        ("こんにちは世界".into(), "こんにちは世界".into()),
        ("こんにちは".into(), "さようなら".into()),
        ("café résumé naïve".into(), "cafe resume naive".into()),
        ("北京是中国的首都".into(), "北京是中华人民共和国的首都".into()),
        ("München ist schön".into(), "Munich is beautiful".into()),
        ("日本語のテスト".into(), "日本語のテスト".into()),
        // Mixed case
        ("Hello World".into(), "hello world".into()),
        ("UPPER CASE TEXT".into(), "upper case text".into()),
        ("CamelCase".into(), "camelcase".into()),
        // Punctuation heavy
        ("Hello, world! How are you?".into(), "Hello world How are you".into()),
        ("test@example.com".into(), "test@example.com".into()),
        // Numbers
        ("The answer is 42".into(), "The answer is 42".into()),
        ("3.14159265".into(), "3.14159".into()),
        // Repeated tokens
        ("the the the the the".into(), "the the the".into()),
        // Very different
        ("alpha beta gamma".into(), "one two three four five six".into()),
        // Whitespace variations
        ("hello   world".into(), "hello world".into()),
    ];
    for (c, r) in &edge_pairs {
        corpus.push(TaggedPair {
            pair: ScoringPair { candidate: c.clone(), reference: r.clone() },
            category: "edge_cases",
        });
    }

    corpus
}

// ═══════════════════════════════════════════════════════════════════════════
// Scoring helpers
// ═══════════════════════════════════════════════════════════════════════════

struct ScoredEntry {
    score: f64,
    agreed: bool,
    time_us: f64,
}

fn score_exact_match_pairs(pairs: &[ScoringPair]) -> Vec<ScoredEntry> {
    let scorer = ExactMatchScorer::case_sensitive();
    pairs.iter().map(|p| {
        let start = Instant::now();
        let result = scorer.score_and_verify(p);
        let elapsed = start.elapsed().as_secs_f64() * 1_000_000.0;
        ScoredEntry {
            score: if result.reference { 1.0 } else { 0.0 },
            agreed: result.agreement,
            time_us: elapsed,
        }
    }).collect()
}

fn score_token_f1_pairs(pairs: &[ScoringPair]) -> Vec<ScoredEntry> {
    let scorer = TokenF1Scorer::default_scorer();
    pairs.iter().map(|p| {
        let start = Instant::now();
        let result = scorer.score_and_verify(p);
        let elapsed = start.elapsed().as_secs_f64() * 1_000_000.0;
        ScoredEntry {
            score: result.reference.f1,
            agreed: result.agreement,
            time_us: elapsed,
        }
    }).collect()
}

fn score_bleu_pairs(pairs: &[ScoringPair]) -> Vec<ScoredEntry> {
    let scorer = BleuScorer::with_smoothing(SmoothingMethod::Add1);
    pairs.iter().map(|p| {
        let start = Instant::now();
        let result = scorer.score_and_verify(p);
        let elapsed = start.elapsed().as_secs_f64() * 1_000_000.0;
        ScoredEntry {
            score: result.reference.score,
            agreed: result.agreement,
            time_us: elapsed,
        }
    }).collect()
}

fn score_rouge1_pairs(pairs: &[ScoringPair]) -> Vec<ScoredEntry> {
    let scorer = RougeNScorer::rouge1();
    pairs.iter().map(|p| {
        let start = Instant::now();
        let result = scorer.score_and_verify(p);
        let elapsed = start.elapsed().as_secs_f64() * 1_000_000.0;
        ScoredEntry {
            score: result.reference.f1,
            agreed: result.agreement,
            time_us: elapsed,
        }
    }).collect()
}

fn score_rougel_pairs(pairs: &[ScoringPair]) -> Vec<ScoredEntry> {
    let scorer = RougeLScorer::default_scorer();
    pairs.iter().map(|p| {
        let start = Instant::now();
        let result = scorer.score_and_verify(p);
        let elapsed = start.elapsed().as_secs_f64() * 1_000_000.0;
        ScoredEntry {
            score: result.reference.f1,
            agreed: result.agreement,
            time_us: elapsed,
        }
    }).collect()
}

fn score_chrf_pairs(pairs: &[ScoringPair]) -> Vec<ScoredEntry> {
    let scorer = ChrFScorer::new(ChrFConfig::default());
    pairs.iter().map(|p| {
        let start = Instant::now();
        let result = scorer.score_and_verify(p);
        let elapsed = start.elapsed().as_secs_f64() * 1_000_000.0;
        ScoredEntry {
            score: result.reference,
            agreed: result.agreement,
            time_us: elapsed,
        }
    }).collect()
}

fn summarize_entries(entries: &[ScoredEntry]) -> (f64, f64, f64, f64, usize, usize, f64) {
    let n = entries.len();
    if n == 0 {
        return (0.0, 0.0, 0.0, 0.0, 0, 0, 0.0);
    }
    let scores: Vec<f64> = entries.iter().map(|e| e.score).collect();
    let mean = scores.iter().sum::<f64>() / n as f64;
    let min = scores.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / n as f64;
    let std_dev = variance.sqrt();
    let agreements = entries.iter().filter(|e| e.agreed).count();
    let disagreements = n - agreements;
    let total_time_ms = entries.iter().map(|e| e.time_us).sum::<f64>() / 1000.0;
    (mean, min, max, std_dev, agreements, disagreements, total_time_ms)
}

// ═══════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════

fn main() {
    let total_start = Instant::now();

    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!("  Spectacles Comprehensive Benchmark");
    eprintln!("═══════════════════════════════════════════════════════════════");

    // Phase 0: Build corpus
    eprintln!("\n▸ Phase 0: Building diverse evaluation corpus...");
    let gen_start = Instant::now();
    let corpus = build_corpus();
    let gen_ms = gen_start.elapsed().as_secs_f64() * 1000.0;

    let categories: Vec<&str> = vec!["mmlu_qa", "squad_qa", "translation", "code_completion", "edge_cases"];
    let corpus_description = CorpusDescription {
        categories: categories.iter().map(|c| {
            let count = corpus.iter().filter(|t| t.category == *c).count();
            CategoryInfo {
                name: c.to_string(),
                count,
                description: match *c {
                    "mmlu_qa" => "MMLU-style multiple-choice Q&A".to_string(),
                    "squad_qa" => "SQuAD-style extractive question answering".to_string(),
                    "translation" => "Translation / paraphrase quality".to_string(),
                    "code_completion" => "Code completion (HumanEval-style)".to_string(),
                    "edge_cases" => "Edge cases: empty, single token, long, unicode".to_string(),
                    _ => String::new(),
                },
            }
        }).collect(),
        total_pairs: corpus.len(),
    };

    eprintln!("  Corpus: {} total pairs", corpus.len());
    for ci in &corpus_description.categories {
        eprintln!("    {:20} {:>4} pairs", ci.name, ci.count);
    }

    // Phase 1: Triple verification scoring
    eprintln!("\n▸ Phase 1: Running triple verification on all metrics...");
    let scoring_start = Instant::now();

    let all_pairs: Vec<ScoringPair> = corpus.iter().map(|t| t.pair.clone()).collect();
    let metric_names = ["exact_match", "token_f1", "bleu", "rouge1", "rouge_l", "chrf"];

    let all_entries: Vec<(&str, Vec<ScoredEntry>)> = vec![
        ("exact_match", score_exact_match_pairs(&all_pairs)),
        ("token_f1", score_token_f1_pairs(&all_pairs)),
        ("bleu", score_bleu_pairs(&all_pairs)),
        ("rouge1", score_rouge1_pairs(&all_pairs)),
        ("rouge_l", score_rougel_pairs(&all_pairs)),
        ("chrf", score_chrf_pairs(&all_pairs)),
    ];

    let scoring_ms = scoring_start.elapsed().as_secs_f64() * 1000.0;

    // Build per-metric summaries
    let mut metric_results = Vec::new();
    let mut per_metric_timing = Vec::new();
    let mut total_disagreements = 0usize;
    let mut total_triple_checks = 0usize;

    eprintln!("\n  {:>12} {:>8} {:>8} {:>8} {:>8} {:>10} {:>8}",
        "Metric", "Mean", "Min", "Max", "StdDev", "Agreement", "Time(ms)");
    eprintln!("  {}", "-".repeat(78));

    for (name, entries) in &all_entries {
        let (mean, min, max, std_dev, agreements, disagreements, time_ms) = summarize_entries(entries);
        let n = entries.len();
        let rate = if n > 0 { agreements as f64 / n as f64 } else { 1.0 };
        total_disagreements += disagreements;
        total_triple_checks += n;

        eprintln!("  {:>12} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>9.1}% {:>8.2}",
            name, mean, min, max, std_dev, rate * 100.0, time_ms);

        metric_results.push(MetricBenchmark {
            metric_name: name.to_string(),
            total_pairs: n,
            mean_score: mean,
            min_score: min,
            max_score: max,
            std_dev,
            triple_agreements: agreements,
            triple_disagreements: disagreements,
            agreement_rate: rate,
            total_time_ms: time_ms,
            per_pair_time_us: if n > 0 { time_ms * 1000.0 / n as f64 } else { 0.0 },
        });
        per_metric_timing.push((name.to_string(), time_ms));
    }

    // Phase 1b: Per-category breakdown
    eprintln!("\n▸ Phase 1b: Per-category breakdown...");
    let mut per_category_results = Vec::new();
    for cat in &categories {
        let cat_indices: Vec<usize> = corpus.iter().enumerate()
            .filter(|(_, t)| t.category == *cat)
            .map(|(i, _)| i)
            .collect();
        let n = cat_indices.len();

        let mut cat_metrics = Vec::new();
        for (mname, entries) in &all_entries {
            let cat_entries: Vec<&ScoredEntry> = cat_indices.iter().map(|&i| &entries[i]).collect();
            let mean = if n > 0 {
                cat_entries.iter().map(|e| e.score).sum::<f64>() / n as f64
            } else { 0.0 };
            let agreements = cat_entries.iter().filter(|e| e.agreed).count();
            let rate = if n > 0 { agreements as f64 / n as f64 } else { 1.0 };
            let time = cat_entries.iter().map(|e| e.time_us).sum::<f64>() / 1000.0;
            cat_metrics.push(CategoryMetricResult {
                metric: mname.to_string(),
                mean_score: mean,
                agreement_rate: rate,
                time_ms: time,
            });
        }
        per_category_results.push(CategoryResults {
            category: cat.to_string(),
            num_pairs: n,
            metrics: cat_metrics,
        });
    }

    // Phase 2: Differential testing
    eprintln!("\n▸ Phase 2: Differential testing (standard + 200 random pairs)...");
    let diff_start = Instant::now();
    let tester = DifferentialTester::new();
    let standard = standard_test_suite();
    let random = random_test_pairs(200, 42);
    let diff_pairs: Vec<ScoringPair> = standard.iter().chain(random.iter()).cloned().collect();
    let num_standard = standard.len();
    let num_random = random.len();

    let em_report = tester.test_exact_match(&diff_pairs);
    let f1_report = tester.test_token_f1(&diff_pairs);
    let bleu_report = tester.test_bleu(&diff_pairs);
    let r1_report = tester.test_rouge1(&diff_pairs);
    let rl_report = tester.test_rouge_l(&diff_pairs);

    let diff_metrics = vec![
        DiffMetricSummary { metric: "exact_match".into(), total_tests: em_report.total_tests, agreements: em_report.agreements, disagreements: em_report.disagreements, agreement_rate: em_report.agreement_rate },
        DiffMetricSummary { metric: "token_f1".into(), total_tests: f1_report.total_tests, agreements: f1_report.agreements, disagreements: f1_report.disagreements, agreement_rate: f1_report.agreement_rate },
        DiffMetricSummary { metric: "bleu".into(), total_tests: bleu_report.total_tests, agreements: bleu_report.agreements, disagreements: bleu_report.disagreements, agreement_rate: bleu_report.agreement_rate },
        DiffMetricSummary { metric: "rouge1".into(), total_tests: r1_report.total_tests, agreements: r1_report.agreements, disagreements: r1_report.disagreements, agreement_rate: r1_report.agreement_rate },
        DiffMetricSummary { metric: "rouge_l".into(), total_tests: rl_report.total_tests, agreements: rl_report.agreements, disagreements: rl_report.disagreements, agreement_rate: rl_report.agreement_rate },
    ];
    let diff_total_checks: usize = diff_metrics.iter().map(|m| m.total_tests).sum();
    let diff_total_disagree: usize = diff_metrics.iter().map(|m| m.disagreements).sum();
    let diff_ms = diff_start.elapsed().as_secs_f64() * 1000.0;

    for dm in &diff_metrics {
        eprintln!("    {:12} {}/{} agree ({:.1}%)",
            dm.metric, dm.agreements, dm.total_tests, dm.agreement_rate * 100.0);
    }

    let differential_test_results = DifferentialTestSummary {
        num_standard_pairs: num_standard,
        num_random_pairs: num_random,
        per_metric: diff_metrics,
        total_checks: diff_total_checks,
        total_disagreements: diff_total_disagree,
    };

    // Phase 3: STARK certification on representative inputs
    eprintln!("\n▸ Phase 3: STARK certification pipeline...");
    let cert_start = Instant::now();

    let cert_metrics = ["exact_match", "token_f1", "bleu", "rouge_1", "chrf"];
    let cert_pairs: Vec<(&str, &str)> = vec![
        ("the cat sat on the mat", "the cat sat on the mat"),
        ("hello world", "hello there world"),
        ("machine learning is great", "deep learning is wonderful"),
        ("Paris", "Paris"),
        ("alpha beta gamma", "alpha gamma delta"),
    ];

    let mut certification_results = Vec::new();
    for metric in &cert_metrics {
        for (cand, refe) in &cert_pairs {
            let preview_len = 50;
            let cand_preview: String = cand.chars().take(preview_len).collect();
            let ref_preview: String = refe.chars().take(preview_len).collect();

            match certify_metric(metric, cand, refe) {
                Ok(cert) => {
                    eprintln!("    {:12} | score={:.4} | proof={} | verified={} | agree={} | prove={:.1}ms | verify={:.1}ms",
                        metric, cert.score, cert.proof_generated, cert.proof_verified,
                        cert.triple_agreement, cert.prove_time_ms, cert.verify_time_ms);
                    certification_results.push(CertificationResult {
                        metric: metric.to_string(),
                        candidate_preview: cand_preview,
                        reference_preview: ref_preview,
                        score: cert.score,
                        proof_generated: cert.proof_generated,
                        proof_verified: cert.proof_verified,
                        triple_agreement: cert.triple_agreement,
                        prove_time_ms: cert.prove_time_ms,
                        verify_time_ms: cert.verify_time_ms,
                        proof_size_bytes: cert.proof_size_bytes,
                        num_wfa_states: cert.num_wfa_states,
                        num_constraints: cert.num_constraints,
                        trace_length: cert.trace_length,
                    });
                }
                Err(e) => {
                    eprintln!("    {:12} | ERROR: {}", metric, e);
                    certification_results.push(CertificationResult {
                        metric: metric.to_string(),
                        candidate_preview: cand_preview,
                        reference_preview: ref_preview,
                        score: 0.0,
                        proof_generated: false,
                        proof_verified: false,
                        triple_agreement: false,
                        prove_time_ms: 0.0,
                        verify_time_ms: 0.0,
                        proof_size_bytes: 0,
                        num_wfa_states: 0,
                        num_constraints: 0,
                        trace_length: 0,
                    });
                }
            }
        }
    }

    let cert_ms = cert_start.elapsed().as_secs_f64() * 1000.0;
    let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;

    // Build final results
    let total_certs = certification_results.len();
    let certs_verified = certification_results.iter().filter(|c| c.proof_verified).count();
    let overall_rate = if total_triple_checks > 0 {
        (total_triple_checks - total_disagreements) as f64 / total_triple_checks as f64
    } else { 1.0 };

    let summary = Summary {
        total_input_pairs: corpus.len(),
        total_metrics: metric_names.len(),
        total_triple_checks,
        total_disagreements,
        overall_agreement_rate: overall_rate,
        all_metrics_agree: total_disagreements == 0,
        total_certifications: total_certs,
        certifications_verified: certs_verified,
        total_wall_clock_ms: total_ms,
    };

    let timing = TimingBreakdown {
        total_ms,
        corpus_generation_ms: gen_ms,
        triple_scoring_ms: scoring_ms,
        certification_ms: cert_ms,
        differential_testing_ms: diff_ms,
        per_metric_ms: per_metric_timing,
    };

    eprintln!("\n═══════════════════════════════════════════════════════════════");
    eprintln!("  Summary");
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!("  Total input pairs:       {}", summary.total_input_pairs);
    eprintln!("  Total metrics:           {}", summary.total_metrics);
    eprintln!("  Total triple checks:     {}", summary.total_triple_checks);
    eprintln!("  Total disagreements:     {}", summary.total_disagreements);
    eprintln!("  Overall agreement rate:  {:.2}%", summary.overall_agreement_rate * 100.0);
    eprintln!("  STARK certifications:    {}/{} verified", summary.certifications_verified, summary.total_certifications);
    eprintln!("  Wall clock time:         {:.1} ms", summary.total_wall_clock_ms);

    let results = ComprehensiveResults {
        timestamp: chrono::Utc::now().to_rfc3339(),
        summary,
        corpus_description,
        metric_results,
        per_category_results,
        certification_results,
        differential_test_results,
        timing,
    };

    let json = serde_json::to_string_pretty(&results).expect("serialize results");

    // Write to file
    let output_path = "comprehensive_benchmark_results.json";
    std::fs::write(output_path, &json).expect("write results file");
    eprintln!("\n  Results saved to: {}", output_path);

    // Write to stdout
    println!("{}", json);
}
