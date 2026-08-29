//! Seq2seq playground tasks + word vocabulary.
//!
//! Translation ru→en is the canonical attention demo: cross-attention learns
//! the word alignment between the two languages. The toy grammar has one
//! twist — Russian adjective/noun order is shuffled while English is fixed —
//! so the learned alignment is a swap, not a trivial diagonal.

use rand::Rng;

use crate::config::TransformerConfig;

/// Russian-side ids share the English word index: `id < EN_BASE` is Russian,
/// `id >= EN_BASE` is its English counterpart.
pub const NOUNS_RU: &[&str] = &["кот", "пёс", "дом", "луна"];
pub const ADJS_RU: &[&str] = &["красный", "синий", "быстрый", "тихий"];
pub const VERBS_RU: &[&str] = &["спит", "бежит"];
pub const EN_BASE: usize = 10;
/// Total data vocabulary (Russian + English words).
pub const VOCAB: usize = 20;

const NOUN_BASE: usize = 0;
const ADJ_BASE: usize = 4;
const VERB_BASE: usize = 8;

fn en(id: usize) -> usize {
    EN_BASE + id
}

fn en_label(ru: &str) -> &'static str {
    match ru {
        "кот" => "cat",
        "пёс" => "dog",
        "дом" => "house",
        "луна" => "moon",
        "красный" => "red",
        "синий" => "blue",
        "быстрый" => "fast",
        "тихий" => "quiet",
        "спит" => "sleeps",
        "бежит" => "runs",
        _ => unreachable!("unknown ru word"),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TransformerTask {
    Translate,
    Reverse,
}

impl TransformerTask {
    pub fn from_id(id: &str) -> Option<Self> {
        match id {
            "translate" => Some(Self::Translate),
            "reverse" => Some(Self::Reverse),
            _ => None,
        }
    }

    pub fn id(&self) -> &'static str {
        match self {
            Self::Translate => "translate",
            Self::Reverse => "reverse",
        }
    }
}

pub struct Sample {
    pub input: Vec<usize>,
    pub target: Vec<usize>,
}

/// `["кот","пёс",…,"cat","dog",…,"<s>","</s>"]` — labels indexed by token id.
pub fn vocab_labels() -> Vec<String> {
    let ru: Vec<&str> = NOUNS_RU
        .iter()
        .chain(ADJS_RU.iter())
        .chain(VERBS_RU.iter())
        .copied()
        .collect();
    let mut labels: Vec<String> = ru.iter().map(|s| s.to_string()).collect();
    labels.extend(ru.iter().map(|s| en_label(s).to_string()));
    labels.push("<s>".to_string());
    labels.push("</s>".to_string());
    labels
}

pub fn sample_sequence<R: Rng>(
    task: TransformerTask,
    _config: &TransformerConfig,
    rng: &mut R,
) -> Sample {
    match task {
        TransformerTask::Translate => {
            let noun = NOUN_BASE + rng.gen_range(0..NOUNS_RU.len());
            let adj = ADJ_BASE + rng.gen_range(0..ADJS_RU.len());
            // Russian mixes adjective/noun order; English is always adj-first.
            let mut input = if rng.gen_bool(0.5) {
                vec![noun, adj]
            } else {
                vec![adj, noun]
            };
            let mut target = vec![en(adj), en(noun)];
            if rng.gen_bool(0.5) {
                let verb = VERB_BASE + rng.gen_range(0..VERBS_RU.len());
                target.push(en(verb));
                // Verb placement also varies on the Russian side.
                if rng.gen_bool(0.5) {
                    input.insert(0, verb);
                } else {
                    input.push(verb);
                }
            }
            Sample { input, target }
        }
        TransformerTask::Reverse => {
            // Distinct Russian words, so the reversed target is unambiguous.
            let len = rng.gen_range(2..=4);
            let mut pool: Vec<usize> = (0..EN_BASE).collect();
            let mut input = Vec::with_capacity(len);
            for _ in 0..len {
                let k = rng.gen_range(0..pool.len());
                input.push(pool.remove(k));
            }
            let mut target = input.clone();
            target.reverse();
            Sample { input, target }
        }
    }
}
