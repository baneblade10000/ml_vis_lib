//! Dev tool: train the reverse task natively, print the loss trajectory.

use rand::rngs::StdRng;
use rand::SeedableRng;
use transformer::config::TransformerConfig;
use transformer::model::{ForwardInput, Model};
use transformer::tasks::{sample_sequence, TransformerTask};


fn labels_for(ids: &[usize]) -> Vec<String> {
    let l = transformer::tasks::vocab_labels();
    ids.iter().map(|i| l[*i].clone()).collect()
}

fn main() {
    let task = std::env::args().nth(3).map(|s| TransformerTask::from_id(&s).unwrap_or(TransformerTask::Translate)).unwrap_or(TransformerTask::Translate);
    let lr: f32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.01);
    let steps: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(6000);
    let mut config = TransformerConfig::default();
    config.learning_rate = lr;
    let mut model = Model::new(config.clone());
    let mut rng = StdRng::seed_from_u64(123);
    let mut ema = 0.0f32;
    for step in 1..=steps {
        let sample = sample_sequence(task, &config, &mut rng);
        let input = ForwardInput {
            enc_tokens: { let mut t = sample.input.clone(); t.push(config.eos()); t },
            dec_in_tokens: { let mut t = vec![config.bos()]; t.extend_from_slice(&sample.target); t },
            targets: { let mut t = sample.target.clone(); t.push(config.eos()); t },
        };
        let loss = model.train_step(&input);
        ema = if ema == 0.0 { loss } else { 0.98 * ema + 0.02 * loss };
        if step % 500 == 0 {
            println!("step {step:>6}  ema {ema:.4}");
        }
    }
    let sample = sample_sequence(task, &config, &mut rng);
    let mut enc = sample.input.clone();
    enc.push(config.eos());
    let (tokens, _) = model.greedy_decode(&enc, config.max_len + 1);
    println!("input    {:?}", labels_for(&sample.input));
    println!("target   {:?}", labels_for(&sample.target));
    let labels = transformer::tasks::vocab_labels();
    println!("greedy   {:?}", tokens.iter().map(|t| &labels[*t]).collect::<Vec<_>>());
}
