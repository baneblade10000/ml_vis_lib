//! Numeric gradient check for the manual backward pass.

use rand::SeedableRng;
use transformer::config::TransformerConfig;
use transformer::model::{backward, ForwardInput, Model, Params};

fn tiny_config() -> TransformerConfig {
    TransformerConfig {
        alphabet_size: 4,
        d_model: 8,
        heads: 2,
        enc_layers: 1,
        dec_layers: 1,
        ff_dim: 12,
        max_len: 4,
        learning_rate: 0.0,
        seed: 7,
    }
}

fn sample_input(config: &TransformerConfig) -> ForwardInput {
    // Fixed sample: reverse of [2, 0, 3] — dec_in = <s> + target, targets = target + </s>.
    let enc = vec![2, 0, 3, config.eos()];
    let target = vec![3, 0, 2];
    let mut dec_in = vec![config.bos()];
    dec_in.extend_from_slice(&target);
    let mut targets = target.clone();
    targets.push(config.eos());
    ForwardInput { enc_tokens: enc, dec_in_tokens: dec_in, targets }
}

fn loss_of(params: &Params, config: &TransformerConfig, input: &ForwardInput) -> f32 {
    params.forward(config, input, false).loss
}

#[test]
fn backward_matches_numeric_gradients() {
    let config = tiny_config();
    let model = Model::new(config.clone());
    let input = sample_input(&config);

    // Analytic grads.
    let pass = model.params.forward(&config, &input, false);
    let mut grads = model.params.zeros_like();
    backward(&model.params, &pass, &mut grads);

    // Numeric central differences on a deterministic sample of entries.
    let tensors = model.params.tensors();
    let grad_tensors = grads.tensors();
    let mut checked = 0usize;
    let mut worst = 0.0f32;
    for (ti, t) in tensors.iter().enumerate() {
        let stride = (t.data.len() / 4).max(1);
        for k in (0..t.data.len()).step_by(stride) {
            let analytic = grad_tensors[ti].data[k];
            if analytic.abs() < 1e-3 {
                continue;
            }
            let h = 2e-3f32;
            let orig = t.data[k];
            let mut params2 = model.params.clone();
            params2.tensors_mut()[ti].data[k] = orig + h;
            let plus = loss_of(&params2, &config, &input);
            params2.tensors_mut()[ti].data[k] = orig - h;
            let minus = loss_of(&params2, &config, &input);
            let numeric = (plus - minus) / (2.0 * h);
            let rel = ((numeric - analytic) / analytic.abs()).abs();
            worst = worst.max(rel);
            assert!(
                rel < 0.08,
                "tensor {ti}[{k}]: analytic={analytic:.6} numeric={numeric:.6} rel={rel:.4}"
            );
            checked += 1;
        }
    }
    // The check must actually exercise a meaningful number of parameters.
    assert!(checked >= 120, "only {checked} entries checked");
}

#[test]
fn translate_task_converges() {
    let mut config = TransformerConfig::default();
    config.learning_rate = 0.003;
    let mut model = Model::new(config.clone());
    let mut rng = rand::rngs::StdRng::seed_from_u64(321);
    let mut ema = 0.0f32;
    for _ in 0..12000 {
        let sample = transformer::tasks::sample_sequence(
            transformer::tasks::TransformerTask::Translate,
            &config,
            &mut rng,
        );
        let input = ForwardInput {
            enc_tokens: { let mut t = sample.input.clone(); t.push(config.eos()); t },
            dec_in_tokens: { let mut t = vec![config.bos()]; t.extend_from_slice(&sample.target); t },
            targets: { let mut t = sample.target.clone(); t.push(config.eos()); t },
        };
        let loss = model.train_step(&input);
        ema = if ema == 0.0 { loss } else { 0.98 * ema + 0.02 * loss };
    }
    assert!(ema < 0.1, "translate loss did not converge: {ema}");
    let sample = transformer::tasks::sample_sequence(
        transformer::tasks::TransformerTask::Translate,
        &config,
        &mut rng,
    );
    let mut enc = sample.input.clone();
    enc.push(config.eos());
    let (tokens, _) = model.greedy_decode(&enc, config.max_len + 1);
    assert_eq!(&tokens[..sample.target.len()], &sample.target[..], "translate greedy mismatch");
}

#[test]
fn reverse_task_converges() {
    let config = TransformerConfig::default();
    let mut model = Model::new(config.clone());
    let mut rng = rand::rngs::StdRng::seed_from_u64(123);
    let mut ema = 0.0f32;
    for _ in 0..18000 {
        let sample = transformer::tasks::sample_sequence(
            transformer::tasks::TransformerTask::Reverse,
            &config,
            &mut rng,
        );
        let input = ForwardInput {
            enc_tokens: { let mut t = sample.input.clone(); t.push(config.eos()); t },
            dec_in_tokens: { let mut t = vec![config.bos()]; t.extend_from_slice(&sample.target); t },
            targets: { let mut t = sample.target.clone(); t.push(config.eos()); t },
        };
        let loss = model.train_step(&input);
        ema = if ema == 0.0 { loss } else { 0.98 * ema + 0.02 * loss };
    }
    assert!(ema < 0.1, "loss did not converge: {ema}");

    // Greedy decode must nail a fresh reverse sample.
    let sample = transformer::tasks::sample_sequence(
        transformer::tasks::TransformerTask::Reverse,
        &config,
        &mut rng,
    );
    let mut enc = sample.input.clone();
    enc.push(config.eos());
    let (tokens, _) = model.greedy_decode(&enc, config.max_len + 1);
    assert_eq!(&tokens[..sample.target.len()], &sample.target[..], "greedy decode mismatch");
}
