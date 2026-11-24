//! Seq2seq playground tasks + word vocabulary.
//!
//! Translation ru→en is the canonical attention demo: cross-attention learns
//! the word alignment between the two languages. The toy grammar has a few
//! twists — Russian may put adjectives after the noun and moves the adverb
//! around the verb, while English word order is fixed — so the learned
//! alignment is a permutation, not a trivial diagonal.

use rand::Rng;

use crate::config::TransformerConfig;

// Adjectives are masculine, so the nouns are too.
pub const NOUNS_RU: &[&str] = &["кот", "пёс", "дом", "ключ", "тигр", "стол"];
pub const ADJS_RU: &[&str] = &["красный", "синий", "быстрый", "тихий", "старый", "малый"];
pub const VERBS_RU: &[&str] = &["спит", "бежит", "видит", "любит"];
pub const ADVERBS_RU: &[&str] = &["тихо", "быстро", "медленно", "часто"];
/// Russian-side ids share the English word index: `id < EN_BASE` is Russian,
/// `id >= EN_BASE` is its English counterpart.
pub const EN_BASE: usize = 20;
/// Total data vocabulary (Russian + English words).
pub const VOCAB: usize = 40;

const NOUN_BASE: usize = 0;
const ADJ_BASE: usize = NOUN_BASE + NOUNS_RU.len();
const VERB_BASE: usize = ADJ_BASE + ADJS_RU.len();
const ADVERB_BASE: usize = VERB_BASE + VERBS_RU.len();
const _: () = assert!(ADVERB_BASE + ADVERBS_RU.len() == EN_BASE);

/// Verbs from this offset on take an object.
const TRANSITIVE_FROM: usize = 2;
/// Share of sentences whose verb takes an object.
const TRANSITIVE_SHARE: f64 = 0.6;
/// Share of noun phrases that stack two adjectives.
const TWO_ADJECTIVES: f64 = 0.25;
/// Share of sentences that get an adverb.
const WITH_ADVERB: f64 = 0.6;

fn en(id: usize) -> usize {
    EN_BASE + id
}

fn en_label(ru: &str) -> &'static str {
    match ru {
        "кот" => "cat",
        "пёс" => "dog",
        "дом" => "house",
        "ключ" => "key",
        "тигр" => "tiger",
        "стол" => "table",
        "красный" => "red",
        "синий" => "blue",
        "быстрый" => "fast",
        "тихий" => "quiet",
        "старый" => "old",
        "малый" => "small",
        "спит" => "sleeps",
        "бежит" => "runs",
        "видит" => "sees",
        "любит" => "likes",
        "тихо" => "quietly",
        "быстро" => "quickly",
        "медленно" => "slowly",
        "часто" => "often",
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
        .chain(ADVERBS_RU.iter())
        .copied()
        .collect();
    let mut labels: Vec<String> = ru.iter().map(|s| s.to_string()).collect();
    labels.extend(ru.iter().map(|s| en_label(s).to_string()));
    labels.push("<s>".to_string());
    labels.push("</s>".to_string());
    labels
}

fn pick<R: Rng>(rng: &mut R, words: &[&str], base: usize) -> usize {
    base + rng.gen_range(0..words.len())
}

/// Noun phrase `[adj, adj?, noun]` in English word order.
fn noun_phrase<R: Rng>(rng: &mut R, noun: usize) -> Vec<usize> {
    let mut phrase = vec![pick(rng, ADJS_RU, ADJ_BASE)];
    if rng.gen_bool(TWO_ADJECTIVES) {
        phrase.push(pick(rng, ADJS_RU, ADJ_BASE));
    }
    phrase.push(noun);
    phrase
}

/// The same noun phrase in Russian: with `flip`, the adjectives follow the noun.
fn russian_phrase(phrase: &[usize], flip: bool) -> Vec<usize> {
    if !flip {
        return phrase.to_vec();
    }
    let (noun, adjectives) = phrase.split_last().expect("noun phrase is never empty");
    let mut ru = vec![*noun];
    ru.extend_from_slice(adjectives);
    ru
}

fn translate_sentence<R: Rng>(rng: &mut R) -> Sample {
    // Distinct nouns keep "the cat sees the cat" out of the data.
    let subject_noun = pick(rng, NOUNS_RU, NOUN_BASE);
    let verb_index = if rng.gen_bool(TRANSITIVE_SHARE) {
        TRANSITIVE_FROM + rng.gen_range(0..VERBS_RU.len() - TRANSITIVE_FROM)
    } else {
        rng.gen_range(0..TRANSITIVE_FROM)
    };
    let verb = VERB_BASE + verb_index;
    let subject = noun_phrase(rng, subject_noun);
    let object = (verb_index >= TRANSITIVE_FROM).then(|| {
        let mut nouns: Vec<usize> = (NOUN_BASE..NOUN_BASE + NOUNS_RU.len())
            .filter(|&noun| noun != subject_noun)
            .collect();
        let noun = nouns.swap_remove(rng.gen_range(0..nouns.len()));
        // English drops the adjective as often as it keeps it.
        if rng.gen_bool(0.5) { vec![noun] } else { noun_phrase(rng, noun) }
    });
    let adverb = rng
        .gen_bool(WITH_ADVERB)
        .then(|| pick(rng, ADVERBS_RU, ADVERB_BASE));

    // English word order is fixed: subject, verb, object, adverb.
    let mut target: Vec<usize> = subject.iter().map(|&word| en(word)).collect();
    target.push(en(verb));
    if let Some(object) = &object {
        target.extend(object.iter().map(|&word| en(word)));
    }
    if let Some(adverb) = adverb {
        target.push(en(adverb));
    }

    let mut input = russian_phrase(&subject, rng.gen_bool(0.5));
    let verb_at = input.len();
    input.push(verb);
    if let Some(object) = &object {
        input.extend(russian_phrase(object, rng.gen_bool(0.5)));
    }
    if let Some(adverb) = adverb {
        // Russian moves the adverb: before the verb, right after it, or sentence-final.
        let at = match rng.gen_range(0..3) {
            0 => verb_at,
            1 => verb_at + 1,
            _ => input.len(),
        };
        input.insert(at, adverb);
    }
    Sample { input, target }
}

fn reverse_sentence<R: Rng>(rng: &mut R) -> Sample {
    // Distinct Russian words, so the reversed target is unambiguous.
    let len = rng.gen_range(4..=7);
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

pub fn sample_sequence<R: Rng>(
    task: TransformerTask,
    _config: &TransformerConfig,
    rng: &mut R,
) -> Sample {
    match task {
        TransformerTask::Translate => translate_sentence(rng),
        TransformerTask::Reverse => reverse_sentence(rng),
    }
}
