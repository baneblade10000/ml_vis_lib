//! Training engine: sampling, stepping, EMA loss, display snapshot.

use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::config::TransformerConfig;
use crate::model::{ForwardInput, Model};
use crate::snapshot::{LossPoint, SnapshotDto};
use crate::tasks::{sample_sequence, Sample, TransformerTask, VOCAB};

pub struct PlaygroundEngine {
    pub config: TransformerConfig,
    pub task: TransformerTask,
    model: Model,
    rng: StdRng,
    step: u64,
    /// Exponential moving average of the per-step training loss.
    ema_loss: f32,
    display_sample: Sample,
    snapshot: Option<SnapshotDto>,
    loss_history: Vec<LossPoint>,
}

impl PlaygroundEngine {
    pub fn new(mut config: TransformerConfig, task: TransformerTask) -> Self {
        // The vocabulary is defined by the task module, not by config JSON.
        config.alphabet_size = VOCAB;
        let rng = StdRng::seed_from_u64(config.seed ^ 0x9e3779b9);
        let model = Model::new(config.clone());
        let mut engine = Self {
            config,
            task,
            model,
            rng,
            step: 0,
            ema_loss: 0.0,
            display_sample: Sample { input: vec![], target: vec![] },
            snapshot: None,
            loss_history: Vec::new(),
        };
        engine.display_sample = engine.fresh_sample();
        engine.refresh_snapshot();
        engine
    }

    fn fresh_sample(&mut self) -> Sample {
        sample_sequence(self.task, &self.config, &mut self.rng)
    }

    /// Rebuilds weights + display sample from scratch.
    pub fn reset(&mut self, task: TransformerTask) {
        self.task = task;
        self.model = Model::new(self.config.clone());
        self.rng = StdRng::seed_from_u64(self.config.seed ^ 0x9e3779b9);
        self.step = 0;
        self.ema_loss = 0.0;
        self.loss_history.clear();
        self.display_sample = self.fresh_sample();
        self.snapshot = None;
        self.refresh_snapshot();
    }

    pub fn set_learning_rate(&mut self, lr: f32) {
        self.config.learning_rate = lr;
        self.model.set_learning_rate(lr);
    }

    /// Draws a fresh display sample without touching the weights.
    pub fn reroll_sample(&mut self) {
        self.display_sample = self.fresh_sample();
        self.refresh_snapshot();
    }

    fn train_input(&self, sample: &Sample) -> ForwardInput {
        ForwardInput {
            enc_tokens: {
                let mut t = sample.input.clone();
                t.push(self.config.eos());
                t
            },
            dec_in_tokens: {
                let mut t = vec![self.config.bos()];
                t.extend_from_slice(&sample.target);
                t
            },
            targets: {
                let mut t = sample.target.clone();
                t.push(self.config.eos());
                t
            },
        }
    }

    /// Runs `n` Adam steps on freshly drawn samples; returns the last loss.
    pub fn train_steps(&mut self, n: usize) -> f32 {
        let mut loss = self.ema_loss;
        for _ in 0..n {
            let sample = self.fresh_sample();
            let input = self.train_input(&sample);
            loss = self.model.train_step(&input);
            self.ema_loss = if self.ema_loss == 0.0 { loss } else { 0.95 * self.ema_loss + 0.05 * loss };
            self.step += 1;
            if self.step % 10 == 0 {
                self.loss_history.push(LossPoint { step: self.step, loss: self.ema_loss });
            }
        }
        loss
    }

    pub fn get_step(&self) -> u64 {
        self.step
    }

    pub fn get_loss(&self) -> f32 {
        self.ema_loss
    }

    /// Recomputes the display snapshot (greedy decode + attention capture).
    pub fn refresh_snapshot(&mut self) {
        let eos = self.config.eos();
        let input = self.display_sample.input.clone();
        let target = self.display_sample.target.clone();
        let mut enc_tokens = input.clone();
        enc_tokens.push(eos);
        let (tokens, pass) = self.model.greedy_decode(&enc_tokens, target.len() + 1);
        let hits = target
            .iter()
            .enumerate()
            .filter(|(i, t)| tokens.get(*i) == Some(*t))
            .count();
        let accuracy = if target.is_empty() { 0.0 } else { hits as f32 / target.len() as f32 };
        let mut dec_in = vec![self.config.bos()];
        dec_in.extend_from_slice(&target);
        let mut enc_in = input.clone();
        enc_in.push(eos);
        self.snapshot = Some(SnapshotDto {
            step: self.step,
            loss: self.ema_loss,
            accuracy,
            task: self.task.id().to_string(),
            input_tokens: input,
            target_tokens: target,
            predicted_tokens: tokens,
            dec_in_tokens: dec_in,
            enc_in_tokens: enc_in,
            attention: (&pass.attention).into(),
            loss_history: self.loss_history.clone(),
            alphabet_size: self.config.alphabet_size,
            labels: crate::tasks::vocab_labels(),
        });
    }

    pub fn snapshot(&mut self) -> &SnapshotDto {
        if self.snapshot.is_none() {
            self.refresh_snapshot();
        }
        self.snapshot.as_ref().unwrap()
    }
}
