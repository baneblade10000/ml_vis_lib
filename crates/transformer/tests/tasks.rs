//! Sampled sentences must fit the model's position table, and a translation
//! must be a permutation of the same words on both sides.

use std::collections::{BTreeMap, BTreeSet};

use rand::{rngs::StdRng, SeedableRng};
use transformer::config::TransformerConfig;
use transformer::tasks::{
    sample_sequence, vocab_labels, Sample, TransformerTask, ADJS_RU, EN_BASE, NOUNS_RU, VERBS_RU,
    VOCAB,
};
use transformer::PlaygroundEngine;

const TASKS: [TransformerTask; 2] = [TransformerTask::Translate, TransformerTask::Reverse];

fn samples(task: TransformerTask, count: usize) -> Vec<Sample> {
    let config = TransformerConfig::default();
    let mut rng = StdRng::seed_from_u64(7);
    (0..count)
        .map(|_| sample_sequence(task, &config, &mut rng))
        .collect()
}

/// The meaning behind a token id: the shared Russian/English word index.
fn meaning(token: usize) -> usize {
    if token >= EN_BASE { token - EN_BASE } else { token }
}

fn is_noun(token: usize) -> bool {
    meaning(token) < NOUNS_RU.len()
}

#[test]
fn samples_fit_the_position_table() {
    let config = TransformerConfig::default();
    for task in TASKS {
        for sample in samples(task, 5000) {
            // The encoder sees input + </s>; the decoder <s> + target + one decoded token.
            assert!(sample.input.len() + 1 <= config.max_len + 2, "{task:?} input too long");
            assert!(sample.target.len() + 2 <= config.max_len + 2, "{task:?} target too long");
        }
    }
}

#[test]
fn tokens_are_in_vocabulary() {
    for task in TASKS {
        for sample in samples(task, 5000) {
            for token in sample.input.iter().chain(&sample.target) {
                assert!(token < &VOCAB, "{task:?}: token {token} outside the vocabulary");
            }
        }
    }
}

#[test]
fn translation_keeps_the_same_words() {
    for sample in samples(TransformerTask::Translate, 5000) {
        let mut source: Vec<usize> = sample.input.iter().map(|&t| meaning(t)).collect();
        let mut translated: Vec<usize> = sample.target.iter().map(|&t| meaning(t)).collect();
        source.sort_unstable();
        translated.sort_unstable();
        assert_eq!(source, translated);
        assert!(sample.input.iter().all(|&t| t < EN_BASE), "Russian side");
        assert!(sample.target.iter().all(|&t| t >= EN_BASE), "English side");
        assert_eq!(sample.input.len(), sample.target.len());
    }
}

#[test]
fn sentences_are_several_words_long() {
    let lengths: Vec<usize> = samples(TransformerTask::Translate, 5000)
        .iter()
        .map(|sample| sample.target.len())
        .collect();
    let mean = lengths.iter().sum::<usize>() as f64 / lengths.len() as f64;
    assert!(mean > 4.0, "mean sentence length {mean:.2}");
    assert!(lengths.iter().all(|&len| (3..=8).contains(&len)), "lengths {lengths:?}");
}

#[test]
fn russian_word_order_is_shuffled() {
    let samples = samples(TransformerTask::Translate, 3000);
    let same_order = samples
        .iter()
        .filter(|sample| {
            let source: Vec<usize> = sample.input.iter().map(|&t| meaning(t)).collect();
            let translated: Vec<usize> = sample.target.iter().map(|&t| meaning(t)).collect();
            source == translated
        })
        .count();
    let shuffled = samples.len() - same_order;
    assert!(shuffled > samples.len() / 2, "only {shuffled} of {} shuffled", samples.len());
    assert!(same_order > 0, "the Russian order is never left alone");
}

#[test]
fn english_word_order_is_fixed() {
    let first_verb = NOUNS_RU.len() + ADJS_RU.len();
    let first_adverb = first_verb + VERBS_RU.len();
    for sample in samples(TransformerTask::Translate, 5000) {
        let mut seen_verb = false;
        for (position, &token) in sample.target.iter().enumerate() {
            let word = meaning(token);
            if word >= first_adverb {
                // An adverb ends the sentence.
                assert_eq!(position, sample.target.len() - 1, "adverb inside {:?}", sample.target);
            } else if word >= first_verb {
                assert!(!seen_verb, "two verbs in {:?}", sample.target);
                seen_verb = true;
            }
        }
        assert!(seen_verb, "no verb in {:?}", sample.target);
    }
}

#[test]
fn a_verb_either_always_or_never_takes_an_object() {
    let first_verb = NOUNS_RU.len() + ADJS_RU.len();
    let mut nouns_by_verb: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
    for sample in samples(TransformerTask::Translate, 5000) {
        let verb = *sample
            .target
            .iter()
            .find(|&&t| meaning(t) >= first_verb)
            .expect("every sentence has a verb");
        let nouns: BTreeSet<usize> = sample.target.iter().copied().filter(|&t| is_noun(t)).collect();
        let count = sample.target.iter().filter(|&&t| is_noun(t)).count();
        assert_eq!(nouns.len(), count, "the same noun twice in {:?}", sample.target);
        nouns_by_verb.entry(verb).or_default().insert(count);
    }
    assert_eq!(nouns_by_verb.len(), VERBS_RU.len());
    for (verb, counts) in &nouns_by_verb {
        assert_eq!(counts.len(), 1, "verb {verb} is inconsistent: {counts:?}");
    }
    let kinds: BTreeSet<usize> = nouns_by_verb.values().map(|counts| *counts.iter().next().unwrap()).collect();
    assert_eq!(kinds, BTreeSet::from([1, 2]), "both verb kinds occur");
}

#[test]
fn reverse_is_the_input_backwards() {
    for sample in samples(TransformerTask::Reverse, 5000) {
        assert!((4..=7).contains(&sample.input.len()));
        // Distinct words, so the reversal is unambiguous.
        let mut words = sample.input.clone();
        words.sort_unstable();
        words.dedup();
        assert_eq!(words.len(), sample.input.len());
        let mut reversed = sample.input.clone();
        reversed.reverse();
        assert_eq!(reversed, sample.target);
    }
}

#[test]
fn labels_cover_the_vocabulary_once() {
    let labels = vocab_labels();
    assert_eq!(labels.len(), VOCAB + 2);
    assert_eq!(labels[VOCAB], "<s>");
    assert_eq!(labels[VOCAB + 1], "</s>");
    let mut sorted = labels.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), labels.len(), "labels repeat");
    assert!(labels[..EN_BASE].iter().all(|label| !label.is_ascii()));
    assert!(labels[EN_BASE..VOCAB].iter().all(|label| label.is_ascii()));
}

#[test]
fn engine_runs_both_tasks_on_the_default_config() {
    for task in TASKS {
        let mut engine = PlaygroundEngine::new(TransformerConfig::default(), task);
        engine.train_steps(5);
        let snapshot = engine.snapshot();
        assert_eq!(snapshot.alphabet_size, VOCAB);
        assert_eq!(snapshot.labels.len(), VOCAB + 2);
        assert_eq!(
            snapshot.enc_in_tokens.len(),
            snapshot.attention.enc_self[0][0][0].len(),
            "encoder rows line up with the encoder input"
        );
    }
}
