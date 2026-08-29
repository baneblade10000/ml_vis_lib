//! Transformer parameters, forward caches, manual backward, Adam.

use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::config::TransformerConfig;
use crate::tensor::{matmul, matmul_at_b, matmul_t_a, Mat};

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

/// Per-head projections. `wq/wk/wv` are `d_model × head_dim`, `wo` is
/// `head_dim × d_model`; the multi-head output is the sum over heads, which is
/// algebraically the usual concat-then-W_O.
#[derive(Clone, Debug)]
pub struct MhaParams {
    pub wq: Vec<Mat>,
    pub wk: Vec<Mat>,
    pub wv: Vec<Mat>,
    pub wo: Vec<Mat>,
}

#[derive(Clone, Debug)]
pub struct LnParams {
    pub gain: Mat, // 1 × d_model
    pub bias: Mat, // 1 × d_model
}

#[derive(Clone, Debug)]
pub struct FfnParams {
    pub w1: Mat, // d_model × ff_dim
    pub b1: Mat, // 1 × ff_dim
    pub w2: Mat, // ff_dim × d_model
    pub b2: Mat, // 1 × d_model
}

#[derive(Clone, Debug)]
pub struct EncLayerParams {
    pub self_attn: MhaParams,
    pub self_ln: LnParams,
    pub ffn: FfnParams,
    pub ffn_ln: LnParams,
}

#[derive(Clone, Debug)]
pub struct DecLayerParams {
    pub self_attn: MhaParams,
    pub self_ln: LnParams,
    pub cross_attn: MhaParams,
    pub cross_ln: LnParams,
    pub ffn: FfnParams,
    pub ffn_ln: LnParams,
}

#[derive(Clone, Debug)]
pub struct Params {
    pub embedding: Mat, // vocab × d_model
    pub pos_enc: Mat,   // (max_len + 2) × d_model
    pub enc: Vec<EncLayerParams>,
    pub dec: Vec<DecLayerParams>,
    pub final_ln: LnParams,
    pub output: Mat, // d_model × vocab
}

impl Params {
    pub fn init(config: &TransformerConfig) -> Self {
        fn uniform(rng: &mut StdRng, rows: usize, cols: usize, std: f32) -> Mat {
            Mat::from_fn(rows, cols, |_, _| (rng.gen::<f32>() * 2.0 - 1.0) * std)
        }
        fn mha(rng: &mut StdRng, config: &TransformerConfig) -> MhaParams {
            let d = config.d_model;
            let dh = config.head_dim();
            let std = (1.0 / d as f32).sqrt();
            MhaParams {
                wq: (0..config.heads).map(|_| uniform(rng, d, dh, std)).collect(),
                wk: (0..config.heads).map(|_| uniform(rng, d, dh, std)).collect(),
                wv: (0..config.heads).map(|_| uniform(rng, d, dh, std)).collect(),
                wo: (0..config.heads).map(|_| uniform(rng, dh, d, std)).collect(),
            }
        }
        fn ln(_rng: &mut StdRng, d: usize) -> LnParams {
            LnParams {
                gain: Mat::from_fn(1, d, |_, _| 1.0),
                bias: Mat::zeros(1, d),
            }
        }
        fn ffn(rng: &mut StdRng, config: &TransformerConfig) -> FfnParams {
            let d = config.d_model;
            FfnParams {
                w1: uniform(rng, d, config.ff_dim, (1.0 / d as f32).sqrt()),
                b1: Mat::zeros(1, config.ff_dim),
                w2: uniform(rng, config.ff_dim, d, (1.0 / config.ff_dim as f32).sqrt()),
                b2: Mat::zeros(1, d),
            }
        }

        let mut rng = StdRng::seed_from_u64(config.seed);
        let d = config.d_model;
        Params {
            embedding: uniform(&mut rng, config.vocab_size(), d, 0.5),
            pos_enc: uniform(&mut rng, config.max_len + 2, d, 0.1),
            enc: (0..config.enc_layers)
                .map(|_| EncLayerParams {
                    self_attn: mha(&mut rng, config),
                    self_ln: ln(&mut rng, d),
                    ffn: ffn(&mut rng, config),
                    ffn_ln: ln(&mut rng, d),
                })
                .collect(),
            dec: (0..config.dec_layers)
                .map(|_| DecLayerParams {
                    self_attn: mha(&mut rng, config),
                    self_ln: ln(&mut rng, d),
                    cross_attn: mha(&mut rng, config),
                    cross_ln: ln(&mut rng, d),
                    ffn: ffn(&mut rng, config),
                    ffn_ln: ln(&mut rng, d),
                })
                .collect(),
            final_ln: ln(&mut rng, d),
            output: uniform(&mut rng, d, config.vocab_size(), (1.0 / d as f32).sqrt()),
        }
    }

    /// Same shapes, all zeros — used as the gradient accumulator.
    pub fn zeros_like(&self) -> Self {
        fn zm(m: &Mat) -> Mat {
            Mat::zeros(m.rows, m.cols)
        }
        fn zmha(m: &MhaParams) -> MhaParams {
            MhaParams {
                wq: m.wq.iter().map(zm).collect(),
                wk: m.wk.iter().map(zm).collect(),
                wv: m.wv.iter().map(zm).collect(),
                wo: m.wo.iter().map(zm).collect(),
            }
        }
        fn zln(m: &LnParams) -> LnParams {
            LnParams { gain: zm(&m.gain), bias: zm(&m.bias) }
        }
        fn zffn(m: &FfnParams) -> FfnParams {
            FfnParams { w1: zm(&m.w1), b1: zm(&m.b1), w2: zm(&m.w2), b2: zm(&m.b2) }
        }
        Params {
            embedding: zm(&self.embedding),
            pos_enc: zm(&self.pos_enc),
            enc: self
                .enc
                .iter()
                .map(|l| EncLayerParams {
                    self_attn: zmha(&l.self_attn),
                    self_ln: zln(&l.self_ln),
                    ffn: zffn(&l.ffn),
                    ffn_ln: zln(&l.ffn_ln),
                })
                .collect(),
            dec: self
                .dec
                .iter()
                .map(|l| DecLayerParams {
                    self_attn: zmha(&l.self_attn),
                    self_ln: zln(&l.self_ln),
                    cross_attn: zmha(&l.cross_attn),
                    cross_ln: zln(&l.cross_ln),
                    ffn: zffn(&l.ffn),
                    ffn_ln: zln(&l.ffn_ln),
                })
                .collect(),
            final_ln: zln(&self.final_ln),
            output: zm(&self.output),
        }
    }

    /// Flat view of every trainable tensor in a fixed order (Adam + gradcheck).
    pub fn tensors(&self) -> Vec<&Mat> {
        let mut out = vec![&self.embedding, &self.pos_enc];
        for l in &self.enc {
            out.extend(l.self_attn.wq.iter());
            out.extend(l.self_attn.wk.iter());
            out.extend(l.self_attn.wv.iter());
            out.extend(l.self_attn.wo.iter());
            out.push(&l.self_ln.gain);
            out.push(&l.self_ln.bias);
            out.push(&l.ffn.w1);
            out.push(&l.ffn.b1);
            out.push(&l.ffn.w2);
            out.push(&l.ffn.b2);
            out.push(&l.ffn_ln.gain);
            out.push(&l.ffn_ln.bias);
        }
        for l in &self.dec {
            out.extend(l.self_attn.wq.iter());
            out.extend(l.self_attn.wk.iter());
            out.extend(l.self_attn.wv.iter());
            out.extend(l.self_attn.wo.iter());
            out.push(&l.self_ln.gain);
            out.push(&l.self_ln.bias);
            out.extend(l.cross_attn.wq.iter());
            out.extend(l.cross_attn.wk.iter());
            out.extend(l.cross_attn.wv.iter());
            out.extend(l.cross_attn.wo.iter());
            out.push(&l.cross_ln.gain);
            out.push(&l.cross_ln.bias);
            out.push(&l.ffn.w1);
            out.push(&l.ffn.b1);
            out.push(&l.ffn.w2);
            out.push(&l.ffn.b2);
            out.push(&l.ffn_ln.gain);
            out.push(&l.ffn_ln.bias);
        }
        out.push(&self.final_ln.gain);
        out.push(&self.final_ln.bias);
        out.push(&self.output);
        out
    }

    pub fn tensors_mut(&mut self) -> Vec<&mut Mat> {
        let mut out: Vec<&mut Mat> = vec![&mut self.embedding, &mut self.pos_enc];
        for l in &mut self.enc {
            out.extend(l.self_attn.wq.iter_mut());
            out.extend(l.self_attn.wk.iter_mut());
            out.extend(l.self_attn.wv.iter_mut());
            out.extend(l.self_attn.wo.iter_mut());
            out.push(&mut l.self_ln.gain);
            out.push(&mut l.self_ln.bias);
            out.push(&mut l.ffn.w1);
            out.push(&mut l.ffn.b1);
            out.push(&mut l.ffn.w2);
            out.push(&mut l.ffn.b2);
            out.push(&mut l.ffn_ln.gain);
            out.push(&mut l.ffn_ln.bias);
        }
        for l in &mut self.dec {
            out.extend(l.self_attn.wq.iter_mut());
            out.extend(l.self_attn.wk.iter_mut());
            out.extend(l.self_attn.wv.iter_mut());
            out.extend(l.self_attn.wo.iter_mut());
            out.push(&mut l.self_ln.gain);
            out.push(&mut l.self_ln.bias);
            out.extend(l.cross_attn.wq.iter_mut());
            out.extend(l.cross_attn.wk.iter_mut());
            out.extend(l.cross_attn.wv.iter_mut());
            out.extend(l.cross_attn.wo.iter_mut());
            out.push(&mut l.cross_ln.gain);
            out.push(&mut l.cross_ln.bias);
            out.push(&mut l.ffn.w1);
            out.push(&mut l.ffn.b1);
            out.push(&mut l.ffn.w2);
            out.push(&mut l.ffn.b2);
            out.push(&mut l.ffn_ln.gain);
            out.push(&mut l.ffn_ln.bias);
        }
        out.push(&mut self.final_ln.gain);
        out.push(&mut self.final_ln.bias);
        out.push(&mut self.output);
        out
    }

    pub fn zero_all(&mut self) {
        for t in self.tensors_mut() {
            t.zero();
        }
    }
}

// ---------------------------------------------------------------------------
// Caches
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct MhaCache {
    pub q_in: Mat,  // post-LN queries input
    pub kv_in: Mat, // source of keys/values (equals q_in for self-attention)
    pub q: Vec<Mat>,
    pub k: Vec<Mat>,
    pub v: Vec<Mat>,
    pub probs: Vec<Mat>, // softmax(scores) per head — also the viz source
    pub head: Vec<Mat>,  // probs @ v per head
}

#[derive(Clone, Debug)]
pub struct LnCache {
    pub x: Mat,
    pub xhat: Mat,
    pub out: Mat, // normalized * gain + bias (needed for the output-projection grad)
}

#[derive(Clone, Debug)]
pub struct FfnCache {
    pub x: Mat,
    pub hidden: Mat, // post-relu
}

#[derive(Clone, Debug)]
pub struct EncLayerCache {
    pub self_ln: LnCache,
    pub self_attn: MhaCache,
    pub ffn_ln: LnCache,
    pub ffn: FfnCache,
}

#[derive(Clone, Debug)]
pub struct DecLayerCache {
    pub self_ln: LnCache,
    pub self_attn: MhaCache,
    pub cross_ln: LnCache,
    pub cross_attn: MhaCache,
    pub ffn_ln: LnCache,
    pub ffn: FfnCache,
}

/// Attention probabilities kept for visualisation: `[layer][head]`.
#[derive(Clone, Debug, Default)]
pub struct AttentionRecord {
    pub enc_self: Vec<Vec<Mat>>,
    pub dec_self: Vec<Vec<Mat>>,
    pub cross: Vec<Vec<Mat>>,
}

#[derive(Clone, Debug)]
pub struct ForwardPass {
    pub enc_tokens: Vec<usize>,
    pub dec_in_tokens: Vec<usize>,
    pub targets: Vec<usize>,
    pub enc: Vec<EncLayerCache>,
    pub memory: Mat,
    pub dec: Vec<DecLayerCache>,
    pub final_ln: LnCache,
    pub logits: Mat,
    pub probs: Mat,
    pub loss: f32,
    pub attention: AttentionRecord,
}

pub struct ForwardInput {
    pub enc_tokens: Vec<usize>,
    pub dec_in_tokens: Vec<usize>,
    pub targets: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Elementwise building blocks
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttnMask {
    None,
    /// Decoder self-attention: query i may only see keys j ≤ i.
    Causal,
}

fn softmax_rows(scores: &Mat, mask: AttnMask) -> Mat {
    let allowed = |i: usize, j: usize| match mask {
        AttnMask::None => true,
        AttnMask::Causal => j <= i,
    };
    let mut out = Mat::zeros(scores.rows, scores.cols);
    for i in 0..scores.rows {
        let row = scores.row(i);
        let max = row
            .iter()
            .enumerate()
            .filter(|(j, _)| allowed(i, *j))
            .map(|(_, v)| *v)
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for (j, &v) in row.iter().enumerate() {
            if !allowed(i, j) {
                continue;
            }
            let e = (v - max).exp();
            out.set(i, j, e);
            sum += e;
        }
        for j in 0..scores.cols {
            out.set(i, j, out.get(i, j) / sum);
        }
    }
    out
}

const LN_EPS: f32 = 1e-5;

fn layernorm(x: &Mat, ln: &LnParams) -> (Mat, LnCache) {
    let mut out = Mat::zeros(x.rows, x.cols);
    let mut xhat = Mat::zeros(x.rows, x.cols);
    let n = x.cols as f32;
    for i in 0..x.rows {
        let row = x.row(i);
        let mean = row.iter().sum::<f32>() / n;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
        let std = (var + LN_EPS).sqrt();
        for j in 0..x.cols {
            let xh = (row[j] - mean) / std;
            xhat.set(i, j, xh);
            out.set(i, j, xh * ln.gain.data[j] + ln.bias.data[j]);
        }
    }
    (out.clone(), LnCache { x: x.clone(), xhat, out })
}

fn ffn_forward(x: &Mat, p: &FfnParams) -> (Mat, FfnCache) {
    let mut hidden = matmul(x, &p.w1);
    for i in 0..hidden.rows {
        for j in 0..hidden.cols {
            let v = hidden.get(i, j) + p.b1.data[j];
            hidden.set(i, j, v.max(0.0));
        }
    }
    let mut out = matmul(&hidden, &p.w2);
    for i in 0..out.rows {
        for j in 0..out.cols {
            out.set(i, j, out.get(i, j) + p.b2.data[j]);
        }
    }
    (out, FfnCache { x: x.clone(), hidden })
}

fn embed(params: &Params, tokens: &[usize]) -> Mat {
    let d = params.embedding.cols;
    let mut out = Mat::zeros(tokens.len(), d);
    for (i, &t) in tokens.iter().enumerate() {
        for j in 0..d {
            out.set(i, j, params.embedding.get(t, j) + params.pos_enc.get(i, j));
        }
    }
    out
}

fn mha_forward(p: &MhaParams, q_in: &Mat, kv_in: &Mat, mask: AttnMask, scale: f32) -> (Mat, MhaCache) {
    let heads = p.wq.len();
    let mut out = Mat::zeros(q_in.rows, p.wo[0].cols);
    let mut cache = MhaCache {
        q_in: q_in.clone(),
        kv_in: kv_in.clone(),
        q: Vec::with_capacity(heads),
        k: Vec::with_capacity(heads),
        v: Vec::with_capacity(heads),
        probs: Vec::with_capacity(heads),
        head: Vec::with_capacity(heads),
    };
    for h in 0..heads {
        let q = matmul(q_in, &p.wq[h]);
        let k = matmul(kv_in, &p.wk[h]);
        let v = matmul(kv_in, &p.wv[h]);
        let mut scores = matmul_at_b(&q, &k);
        for val in scores.data.iter_mut() {
            *val *= scale;
        }
        let probs = softmax_rows(&scores, mask);
        let head = matmul(&probs, &v);
        let projected = matmul(&head, &p.wo[h]);
        out.add_assign(&projected);
        cache.q.push(q);
        cache.k.push(k);
        cache.v.push(v);
        cache.probs.push(probs);
        cache.head.push(head);
    }
    (out, cache)
}

// ---------------------------------------------------------------------------
// Forward
// ---------------------------------------------------------------------------

fn encode(
    params: &Params,
    config: &TransformerConfig,
    enc_tokens: &[usize],
    record: bool,
) -> (Mat, Vec<EncLayerCache>, AttentionRecord) {
    let scale = 1.0 / (config.head_dim() as f32).sqrt();
    let mut x = embed(params, enc_tokens);
    let mut caches = Vec::with_capacity(params.enc.len());
    let mut record_attn = AttentionRecord {
        enc_self: Vec::with_capacity(params.enc.len()),
        dec_self: Vec::new(),
        cross: Vec::new(),
    };
    for layer in &params.enc {
        let (normed, self_ln) = layernorm(&x, &layer.self_ln);
        let (attn_out, self_attn) = mha_forward(&layer.self_attn, &normed, &normed, AttnMask::None, scale);
        x.add_assign(&attn_out);
        let (normed2, ffn_ln) = layernorm(&x, &layer.ffn_ln);
        let (ffn_out, ffn) = ffn_forward(&normed2, &layer.ffn);
        x.add_assign(&ffn_out);
        if record {
            record_attn.enc_self.push(self_attn.probs.clone());
        }
        caches.push(EncLayerCache { self_ln, self_attn, ffn_ln, ffn });
    }
    (x, caches, record_attn)
}

impl Params {
    pub fn forward(&self, config: &TransformerConfig, input: &ForwardInput, record: bool) -> ForwardPass {
        let scale = 1.0 / (config.head_dim() as f32).sqrt();
        let (memory, enc_caches, mut attention) = encode(self, config, &input.enc_tokens, record);

        let mut x = embed(self, &input.dec_in_tokens);
        let mut caches = Vec::with_capacity(self.dec.len());
        for layer in &self.dec {
            let (normed, self_ln) = layernorm(&x, &layer.self_ln);
            let (self_out, self_attn) = mha_forward(&layer.self_attn, &normed, &normed, AttnMask::Causal, scale);
            x.add_assign(&self_out);
            let (normed2, cross_ln) = layernorm(&x, &layer.cross_ln);
            let (cross_out, cross_attn) = mha_forward(&layer.cross_attn, &normed2, &memory, AttnMask::None, scale);
            x.add_assign(&cross_out);
            let (normed3, ffn_ln) = layernorm(&x, &layer.ffn_ln);
            let (ffn_out, ffn) = ffn_forward(&normed3, &layer.ffn);
            x.add_assign(&ffn_out);
            if record {
                attention.dec_self.push(self_attn.probs.clone());
                attention.cross.push(cross_attn.probs.clone());
            }
            caches.push(DecLayerCache { self_ln, self_attn, cross_ln, cross_attn, ffn_ln, ffn });
        }

        let (final_norm, final_ln) = layernorm(&x, &self.final_ln);
        let logits = matmul(&final_norm, &self.output);
        let probs = softmax_rows(&logits, AttnMask::None);
        let rows = logits.rows as f32;
        let loss = input
            .targets
            .iter()
            .enumerate()
            .map(|(i, &t)| -probs.get(i, t).ln())
            .sum::<f32>()
            / rows;

        ForwardPass {
            enc_tokens: input.enc_tokens.clone(),
            dec_in_tokens: input.dec_in_tokens.clone(),
            targets: input.targets.clone(),
            enc: enc_caches,
            memory,
            dec: caches,
            final_ln,
            logits,
            probs,
            loss,
            attention,
        }
    }
}

// ---------------------------------------------------------------------------
// Backward
// ---------------------------------------------------------------------------

fn softmax_backward(probs: &Mat, d_out: &Mat) -> Mat {
    let mut d_scores = Mat::zeros(probs.rows, probs.cols);
    for i in 0..probs.rows {
        let p = probs.row(i);
        let dp = d_out.row(i);
        let dot: f32 = p.iter().zip(dp.iter()).map(|(a, b)| a * b).sum();
        for j in 0..probs.cols {
            d_scores.set(i, j, p[j] * (dp[j] - dot));
        }
    }
    d_scores
}

/// Returns d_x; accumulates gain/bias grads into `d_ln`.
fn layernorm_backward(cache: &LnCache, ln: &LnParams, d_ln: &mut LnParams, d_out: &Mat) -> Mat {
    let n = cache.x.cols as f32;
    let mut d_x = Mat::zeros(cache.x.rows, cache.x.cols);
    for i in 0..cache.x.rows {
        let row = cache.x.row(i);
        let mean = row.iter().sum::<f32>() / n;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
        let std = (var + LN_EPS).sqrt();
        let dy = d_out.row(i);
        let xhat = cache.xhat.row(i);
        for j in 0..cache.x.cols {
            d_ln.gain.data[j] += dy[j] * xhat[j];
            d_ln.bias.data[j] += dy[j];
        }
        let mut d_mean = 0.0f32;
        let mut d_var = 0.0f32;
        for j in 0..cache.x.cols {
            let dxhat = dy[j] * ln.gain.data[j];
            d_var += dxhat * (row[j] - mean) * (-1.0 / (2.0 * std * std * std));
            d_mean += dxhat * (-1.0 / std);
        }
        for j in 0..cache.x.cols {
            let dxhat = dy[j] * ln.gain.data[j];
            let dx = dxhat / std + d_var * 2.0 * (row[j] - mean) / n + d_mean / n;
            d_x.set(i, j, dx);
        }
    }
    d_x
}

/// Returns d_x; accumulates weight/bias grads into `d_p`.
fn ffn_backward(cache: &FfnCache, p: &FfnParams, d_p: &mut FfnParams, d_out: &Mat) -> Mat {
    let d_w2 = matmul_t_a(&cache.hidden, d_out); // ff_dim × d_model
    for j in 0..d_w2.cols {
        let mut col_sum = 0.0f32;
        for i in 0..d_out.rows {
            col_sum += d_out.get(i, j);
        }
        d_p.b2.data[j] += col_sum;
    }
    let d_hidden = matmul_at_b(d_out, &p.w2); // T × ff_dim (d_out @ w2ᵀ)
    let mut d_hidden_pre = d_hidden;
    for i in 0..d_hidden_pre.rows {
        for j in 0..d_hidden_pre.cols {
            let keep = cache.hidden.get(i, j) > 0.0;
            if !keep {
                d_hidden_pre.set(i, j, 0.0);
            }
        }
    }
    let d_w1 = matmul_t_a(&cache.x, &d_hidden_pre); // d_model × ff_dim
    for j in 0..d_w1.cols {
        let mut col_sum = 0.0f32;
        for i in 0..d_hidden_pre.rows {
            col_sum += d_hidden_pre.get(i, j);
        }
        d_p.b1.data[j] += col_sum;
    }
    d_p.w1.add_assign(&d_w1);
    d_p.w2.add_assign(&d_w2);
    matmul_at_b(&d_hidden_pre, &p.w1) // T × d_model (d_hidden_pre @ w1ᵀ)
}

/// Returns (d_q_in, d_kv_in); accumulates projection grads into `d_p`.
fn mha_backward(
    cache: &MhaCache,
    p: &MhaParams,
    d_p: &mut MhaParams,
    d_out: &Mat,
    mask: AttnMask,
) -> (Mat, Mat) {
    let _ = mask; // masked probs are zero, so the softmax jacobian zeroes their grads
    let scale = 1.0 / (p.wq[0].cols as f32).sqrt();
    let heads = p.wq.len();
    let mut d_q_in = Mat::zeros(cache.q_in.rows, cache.q_in.cols);
    let mut d_kv_in = Mat::zeros(cache.kv_in.rows, cache.kv_in.cols);
    for h in 0..heads {
        // output projection
        let d_wo = matmul_t_a(&cache.head[h], d_out); // head_dim × d_model
        let d_head = matmul_at_b(d_out, &p.wo[h]); // T × head_dim
        d_p.wo[h].add_assign(&d_wo);
        // attention probabilities
        let d_probs = matmul_at_b(&d_head, &cache.v[h]); // T × T_kv
        let d_v = matmul_t_a(&cache.probs[h], &d_head); // T_kv × head_dim
        let d_scores = softmax_backward(&cache.probs[h], &d_probs);
        let mut d_q = matmul(&d_scores, &cache.k[h]); // T × head_dim
        let mut d_k = matmul_t_a(&d_scores, &cache.q[h]); // T_kv × head_dim
        for v in d_q.data.iter_mut() {
            *v *= scale;
        }
        for v in d_k.data.iter_mut() {
            *v *= scale;
        }
        // input projections
        d_p.wq[h].add_assign(&matmul_t_a(&cache.q_in, &d_q));
        d_p.wk[h].add_assign(&matmul_t_a(&cache.kv_in, &d_k));
        d_p.wv[h].add_assign(&matmul_t_a(&cache.kv_in, &d_v));
        let d_q_path = matmul_at_b(&d_q, &p.wq[h]); // T × d_model
        let d_k_path = matmul_at_b(&d_k, &p.wk[h]);
        let d_v_path = matmul_at_b(&d_v, &p.wv[h]);
        d_q_in.add_assign(&d_q_path);
        d_kv_in.add_assign(&d_k_path);
        d_kv_in.add_assign(&d_v_path);
    }
    (d_q_in, d_kv_in)
}

/// Full backward: seeds from cross-entropy, accumulates into `grads`.
pub fn backward(params: &Params, pass: &ForwardPass, grads: &mut Params) {
    // Cross-entropy seed.
    let rows = pass.logits.rows as f32;
    let mut d_logits = pass.probs.clone();
    for i in 0..d_logits.rows {
        for j in 0..d_logits.cols {
            let onehot = if j == pass.targets[i] { 1.0 } else { 0.0 };
            d_logits.set(i, j, (d_logits.get(i, j) - onehot) / rows);
        }
    }

    // Output head: logits = final_norm @ W_out.
    let d_out_final = matmul_at_b(&d_logits, &params.output); // T × d_model
    let d_w_out = matmul_t_a(&pass.final_ln.out, &d_logits);
    grads.output.add_assign(&d_w_out);
    let mut d_x = layernorm_backward(&pass.final_ln, &params.final_ln, &mut grads.final_ln, &d_out_final);

    // Decoder stack, reverse order.
    let mut d_memory = Mat::zeros(pass.memory.rows, pass.memory.cols);
    for (l, layer) in params.dec.iter().enumerate().rev() {
        let cache = &pass.dec[l];
        // FFN residual block.
        let d_normed3 = ffn_backward(&cache.ffn, &layer.ffn, &mut grads.dec[l].ffn, &d_x);
        let mut d_x2 = d_x.clone();
        d_x2.add_assign(&layernorm_backward(&cache.ffn_ln, &layer.ffn_ln, &mut grads.dec[l].ffn_ln, &d_normed3));
        // Cross-attention block (K/V come from the encoder memory).
        let (d_q2, d_kv_mem) = mha_backward(
            &cache.cross_attn,
            &layer.cross_attn,
            &mut grads.dec[l].cross_attn,
            &d_x2,
            AttnMask::None,
        );
        let mut d_x1 = d_x2.clone();
        d_x1.add_assign(&layernorm_backward(&cache.cross_ln, &layer.cross_ln, &mut grads.dec[l].cross_ln, &d_q2));
        d_memory.add_assign(&d_kv_mem);
        // Masked self-attention block.
        let (d_q1, d_kv1) = mha_backward(
            &cache.self_attn,
            &layer.self_attn,
            &mut grads.dec[l].self_attn,
            &d_x1,
            AttnMask::Causal,
        );
        let mut d_normed1 = d_q1;
        d_normed1.add_assign(&d_kv1);
        let d_x0 = layernorm_backward(&cache.self_ln, &layer.self_ln, &mut grads.dec[l].self_ln, &d_normed1);
        let mut d_x_prev = d_x1;
        d_x_prev.add_assign(&d_x0);
        d_x = d_x_prev;
    }
    scatter_embed_grads(&pass.dec_in_tokens, &d_x, grads);

    // Encoder stack, reverse order, seeded by the memory grad.
    let mut d_xe = d_memory;
    for (l, layer) in params.enc.iter().enumerate().rev() {
        let cache = &pass.enc[l];
        let d_normed2 = ffn_backward(&cache.ffn, &layer.ffn, &mut grads.enc[l].ffn, &d_xe);
        let mut d_x2 = d_xe.clone();
        d_x2.add_assign(&layernorm_backward(&cache.ffn_ln, &layer.ffn_ln, &mut grads.enc[l].ffn_ln, &d_normed2));
        let (d_q, d_kv) = mha_backward(
            &cache.self_attn,
            &layer.self_attn,
            &mut grads.enc[l].self_attn,
            &d_x2,
            AttnMask::None,
        );
        let mut d_normed = d_q;
        d_normed.add_assign(&d_kv);
        let d_x0 = layernorm_backward(&cache.self_ln, &layer.self_ln, &mut grads.enc[l].self_ln, &d_normed);
        let mut d_x_prev = d_x2;
        d_x_prev.add_assign(&d_x0);
        d_xe = d_x_prev;
    }
    scatter_embed_grads(&pass.enc_tokens, &d_xe, grads);
}

fn scatter_embed_grads(tokens: &[usize], d_x: &Mat, grads: &mut Params) {
    for (i, &t) in tokens.iter().enumerate() {
        for j in 0..d_x.cols {
            let dv = d_x.get(i, j);
            let e = grads.embedding.get(t, j) + dv;
            grads.embedding.set(t, j, e);
            let p = grads.pos_enc.get(i, j) + dv;
            grads.pos_enc.set(i, j, p);
        }
    }
}

// ---------------------------------------------------------------------------
// Model: Adam + train step + greedy decode
// ---------------------------------------------------------------------------

pub struct Model {
    pub params: Params,
    pub config: TransformerConfig,
    grads: Params,
    adam_m: Vec<Vec<f32>>,
    adam_v: Vec<Vec<f32>>,
    adam_t: u64,
}

impl Model {
    pub fn new(config: TransformerConfig) -> Self {
        let params = Params::init(&config);
        let grads = params.zeros_like();
        let tensors = params.tensors();
        let adam_m = tensors.iter().map(|t| vec![0.0; t.data.len()]).collect();
        let adam_v = tensors.iter().map(|t| vec![0.0; t.data.len()]).collect();
        Self { params, config, grads, adam_m, adam_v, adam_t: 0 }
    }

    pub fn set_learning_rate(&mut self, lr: f32) {
        self.config.learning_rate = lr;
    }

    pub fn loss_and_grads(&mut self, input: &ForwardInput) -> f32 {
        let pass = self.params.forward(&self.config, input, false);
        self.grads.zero_all();
        backward(&self.params, &pass, &mut self.grads);
        pass.loss
    }

    /// One Adam step on a fresh sample; returns the pre-step loss.
    pub fn train_step(&mut self, input: &ForwardInput) -> f32 {
        let loss = self.loss_and_grads(input);
        self.adam_t += 1;
        let t = self.adam_t as f32;
        let lr = self.config.learning_rate;
        let (b1, b2) = (0.9f32, 0.999f32);
        let c1 = 1.0 - b1.powf(t);
        let c2 = 1.0 - b2.powf(t);
        let mut idx = 0;
        for p in self.params.tensors_mut() {
            let g = &self.grads.tensors()[idx];
            let m = &mut self.adam_m[idx];
            let v = &mut self.adam_v[idx];
            for k in 0..p.data.len() {
                m[k] = b1 * m[k] + (1.0 - b1) * g.data[k];
                v[k] = b2 * v[k] + (1.0 - b2) * g.data[k] * g.data[k];
                let m_hat = m[k] / c1;
                let v_hat = v[k] / c2;
                p.data[k] -= lr * m_hat / (v_hat.sqrt() + 1e-8);
            }
            idx += 1;
        }
        loss
    }

    /// Greedy autoregressive decode; the returned pass records attention for
    /// the full final sequence (teacher-forced re-run of the decoded output).
    pub fn greedy_decode(&self, enc_tokens: &[usize], max_len: usize) -> (Vec<usize>, ForwardPass) {
        let bos = self.config.bos();
        let eos = self.config.eos();
        let mut dec_in = vec![bos];
        let mut tokens: Vec<usize> = Vec::new();
        for _ in 0..max_len {
            let input = ForwardInput {
                enc_tokens: enc_tokens.to_vec(),
                dec_in_tokens: dec_in.clone(),
                targets: vec![0; dec_in.len()],
            };
            let pass = self.params.forward(&self.config, &input, false);
            let last = (pass.logits.rows - 1) * pass.logits.cols;
            let mut best = 0usize;
            for j in 1..pass.logits.cols {
                if pass.logits.data[last + j] > pass.logits.data[last + best] {
                    best = j;
                }
            }
            tokens.push(best);
            if best == eos {
                break;
            }
            dec_in.push(best);
        }
        // Final teacher-forced pass over the whole decoded sequence for viz.
        let full_in = ForwardInput {
            enc_tokens: enc_tokens.to_vec(),
            dec_in_tokens: dec_in.clone(),
            targets: vec![0; dec_in.len()],
        };
        let pass = self.params.forward(&self.config, &full_in, true);
        (tokens, pass)
    }
}
