//! CNN layers with forward / backward (CPU).

use rand::Rng;

use crate::config::{ActivationId, LayerKind, PoolKind, CnnConfig, SIGNAL_LENGTH};
use crate::tensor::{
    act_grad_from_out, apply_act, dense_backward, dense_forward, kaiming_uniform, zeros_signal,
    zeros_volume, Map2D, Signal, Volume,
};

#[derive(Clone, Debug)]
pub struct LayerMeta {
    pub id: String,
    pub kind: String,
    pub label: String,
    pub channels: usize,
    pub rows: usize,
    pub cols: usize,
    pub length: usize,
    pub space: &'static str,
    pub params: usize,
}

pub enum ActCache {
    None,
    Vol(Volume),
    Sig(Signal),
    Flat(Vec<f32>),
}

#[derive(Clone)]
pub enum Layer {
    Input2d {
        h: usize,
        w: usize,
    },
    Input1d {
        len: usize,
    },
    Conv2d {
        filters: usize,
        k: usize,
        in_c: usize,
        act: String,
        /// W[out][in][k][k] flat: out*in*k*k
        w: Vec<f32>,
        b: Vec<f32>,
        gw: Vec<f32>,
        gb: Vec<f32>,
        last_in: Volume,
        last_pre: Volume,
        last_out: Volume,
    },
    Pool2d {
        kind: PoolKind,
        last_in: Volume,
        last_out: Volume,
        /// argmax flat index per output cell for max pool
        switches: Vec<usize>,
    },
    Gap2d {
        last_in: Volume,
        last_out: Signal,
    },
    Flatten {
        c: usize,
        h: usize,
        w: usize,
        last_flat: Vec<f32>,
    },
    Conv1d {
        filters: usize,
        k: usize,
        in_c: usize,
        act: String,
        w: Vec<f32>,
        b: Vec<f32>,
        gw: Vec<f32>,
        gb: Vec<f32>,
        last_in: Signal,
        last_pre: Signal,
        last_out: Signal,
    },
    Pool1d {
        kind: PoolKind,
        last_in: Signal,
        last_out: Signal,
        switches: Vec<usize>,
    },
    Gap1d {
        last_in: Signal,
        last_out: Signal,
    },
    Dense {
        units: usize,
        inn: usize,
        act: String,
        w: Vec<f32>,
        b: Vec<f32>,
        gw: Vec<f32>,
        gb: Vec<f32>,
        last_in: Vec<f32>,
        last_pre: Vec<f32>,
        last_out: Vec<f32>,
    },
    Output {
        inn: usize,
        w: Vec<f32>,
        b: Vec<f32>,
        gw: Vec<f32>,
        gb: Vec<f32>,
        last_in: Vec<f32>,
        last_logit: f32,
    },
}

pub enum Activ {
    Vol(Volume),
    Sig(Signal),
    Flat(Vec<f32>),
    Logit(f32),
}

fn act_name(a: &Option<ActivationId>, cfg: &CnnConfig) -> String {
    let id = a.as_ref().unwrap_or(&cfg.activation);
    match id {
        ActivationId::Relu => "relu",
        ActivationId::Tanh => "tanh",
        ActivationId::Sigmoid => "sigmoid",
        ActivationId::Linear => "linear",
    }
    .into()
}

pub fn build_pipeline(cfg: &CnnConfig, rng: &mut impl Rng) -> (Vec<Layer>, Vec<LayerMeta>) {
    let mut layers = Vec::new();
    let mut metas = Vec::new();
    let mut next_id = 1usize;

    match cfg.mode {
        crate::config::CnnMode::D2 => {
            let size = cfg.image_size();
            let mut c = 1usize;
            let mut h = size;
            let mut w = size;
            let mut flat_len: Option<usize> = None;

            layers.push(Layer::Input2d { h, w });
            metas.push(LayerMeta {
                id: "in".into(),
                kind: "input".into(),
                label: format!("Input {size}×{size}"),
                channels: 1,
                rows: h,
                cols: w,
                length: 0,
                space: "2d",
                params: 0,
            });

            for spec in &cfg.layers {
                let id = format!("L{next_id}");
                next_id += 1;
                match spec.kind {
                    LayerKind::Conv2d => {
                        let filters = spec.filters.unwrap_or(4);
                        let k = spec.kernel_size.unwrap_or(3);
                        let act = act_name(&spec.activation, cfg);
                        let n = filters * c * k * k;
                        let weights = kaiming_uniform(c * k * k, n, rng);
                        let bias = vec![0.0; filters];
                        layers.push(Layer::Conv2d {
                            filters,
                            k,
                            in_c: c,
                            act,
                            gw: vec![0.0; n],
                            gb: vec![0.0; filters],
                            w: weights,
                            b: bias,
                            last_in: vec![],
                            last_pre: vec![],
                            last_out: vec![],
                        });
                        let params = n + filters;
                        c = filters;
                        metas.push(LayerMeta {
                            id,
                            kind: "conv2d".into(),
                            label: format!("Conv {filters}×{k}×{k}"),
                            channels: c,
                            rows: h,
                            cols: w,
                            length: 0,
                            space: "2d",
                            params,
                        });
                    }
                    LayerKind::Pool2d => {
                        let kind = spec.pool_kind.clone().unwrap_or(PoolKind::Max);
                        layers.push(Layer::Pool2d {
                            kind,
                            last_in: vec![],
                            last_out: vec![],
                            switches: vec![],
                        });
                        h = (h / 2).max(1);
                        w = (w / 2).max(1);
                        metas.push(LayerMeta {
                            id,
                            kind: "pool2d".into(),
                            label: "Pool 2×2".into(),
                            channels: c,
                            rows: h,
                            cols: w,
                            length: 0,
                            space: "2d",
                            params: 0,
                        });
                    }
                    LayerKind::Gap2d => {
                        layers.push(Layer::Gap2d {
                            last_in: vec![],
                            last_out: vec![],
                        });
                        flat_len = Some(c);
                        metas.push(LayerMeta {
                            id,
                            kind: "gap2d".into(),
                            label: "GAP".into(),
                            channels: c,
                            rows: 1,
                            cols: 1,
                            length: c,
                            space: "1d",
                            params: 0,
                        });
                    }
                    LayerKind::Flatten => {
                        layers.push(Layer::Flatten {
                            c,
                            h,
                            w,
                            last_flat: vec![],
                        });
                        flat_len = Some(c * h * w);
                        metas.push(LayerMeta {
                            id,
                            kind: "flatten".into(),
                            label: "Flatten".into(),
                            channels: 1,
                            rows: 0,
                            cols: 0,
                            length: flat_len.unwrap(),
                            space: "1d",
                            params: 0,
                        });
                    }
                    LayerKind::Dense => {
                        let units = spec.units.unwrap_or(1);
                        let inn = flat_len.unwrap_or(c * h * w);
                        let act = act_name(&spec.activation, cfg);
                        let n = units * inn;
                        layers.push(Layer::Dense {
                            units,
                            inn,
                            act,
                            w: kaiming_uniform(inn, n, rng),
                            b: vec![0.0; units],
                            gw: vec![0.0; n],
                            gb: vec![0.0; units],
                            last_in: vec![],
                            last_pre: vec![],
                            last_out: vec![],
                        });
                        flat_len = Some(units);
                        metas.push(LayerMeta {
                            id,
                            kind: "dense".into(),
                            label: format!("Dense {units}"),
                            channels: 1,
                            rows: 0,
                            cols: 0,
                            length: units,
                            space: "1d",
                            params: n + units,
                        });
                    }
                    _ => {}
                }
            }

            let inn = flat_len.unwrap_or(c * h * w);
            let id = format!("L{next_id}");
            layers.push(Layer::Output {
                inn,
                w: kaiming_uniform(inn, inn, rng),
                b: vec![0.0],
                gw: vec![0.0; inn],
                gb: vec![0.0],
                last_in: vec![],
                last_logit: 0.0,
            });
            metas.push(LayerMeta {
                id,
                kind: "output".into(),
                label: "Output".into(),
                channels: 1,
                rows: 0,
                cols: 0,
                length: 1,
                space: "1d",
                params: inn + 1,
            });
        }
        crate::config::CnnMode::D1 => {
            let mut c = 1usize;
            let mut len = SIGNAL_LENGTH;
            let mut flat_len: Option<usize> = None;

            layers.push(Layer::Input1d { len });
            metas.push(LayerMeta {
                id: "in".into(),
                kind: "input".into(),
                label: format!("Input {SIGNAL_LENGTH}"),
                channels: 1,
                rows: 0,
                cols: 0,
                length: len,
                space: "1d",
                params: 0,
            });

            for spec in &cfg.layers {
                let id = format!("L{next_id}");
                next_id += 1;
                match spec.kind {
                    LayerKind::Conv1d => {
                        let filters = spec.filters.unwrap_or(4);
                        let k = spec.kernel_size.unwrap_or(5);
                        let act = act_name(&spec.activation, cfg);
                        let n = filters * c * k;
                        layers.push(Layer::Conv1d {
                            filters,
                            k,
                            in_c: c,
                            act,
                            w: kaiming_uniform(c * k, n, rng),
                            b: vec![0.0; filters],
                            gw: vec![0.0; n],
                            gb: vec![0.0; filters],
                            last_in: vec![],
                            last_pre: vec![],
                            last_out: vec![],
                        });
                        let params = n + filters;
                        c = filters;
                        metas.push(LayerMeta {
                            id,
                            kind: "conv1d".into(),
                            label: format!("Conv1d {filters}×{k}"),
                            channels: c,
                            rows: 0,
                            cols: 0,
                            length: len,
                            space: "1d",
                            params,
                        });
                    }
                    LayerKind::Pool1d => {
                        let kind = spec.pool_kind.clone().unwrap_or(PoolKind::Max);
                        layers.push(Layer::Pool1d {
                            kind,
                            last_in: vec![],
                            last_out: vec![],
                            switches: vec![],
                        });
                        len = (len / 2).max(1);
                        metas.push(LayerMeta {
                            id,
                            kind: "pool1d".into(),
                            label: "Pool1d".into(),
                            channels: c,
                            rows: 0,
                            cols: 0,
                            length: len,
                            space: "1d",
                            params: 0,
                        });
                    }
                    LayerKind::Gap1d => {
                        layers.push(Layer::Gap1d {
                            last_in: vec![],
                            last_out: vec![],
                        });
                        flat_len = Some(c);
                        metas.push(LayerMeta {
                            id,
                            kind: "gap1d".into(),
                            label: "GAP1d".into(),
                            channels: c,
                            rows: 0,
                            cols: 0,
                            length: c,
                            space: "1d",
                            params: 0,
                        });
                    }
                    LayerKind::Flatten => {
                        layers.push(Layer::Flatten {
                            c,
                            h: 1,
                            w: len,
                            last_flat: vec![],
                        });
                        flat_len = Some(c * len);
                        metas.push(LayerMeta {
                            id,
                            kind: "flatten".into(),
                            label: "Flatten".into(),
                            channels: 1,
                            rows: 0,
                            cols: 0,
                            length: flat_len.unwrap(),
                            space: "1d",
                            params: 0,
                        });
                    }
                    LayerKind::Dense => {
                        let units = spec.units.unwrap_or(1);
                        let inn = flat_len.unwrap_or(c * len);
                        let act = act_name(&spec.activation, cfg);
                        let n = units * inn;
                        layers.push(Layer::Dense {
                            units,
                            inn,
                            act,
                            w: kaiming_uniform(inn, n, rng),
                            b: vec![0.0; units],
                            gw: vec![0.0; n],
                            gb: vec![0.0; units],
                            last_in: vec![],
                            last_pre: vec![],
                            last_out: vec![],
                        });
                        flat_len = Some(units);
                        metas.push(LayerMeta {
                            id,
                            kind: "dense".into(),
                            label: format!("Dense {units}"),
                            channels: 1,
                            rows: 0,
                            cols: 0,
                            length: units,
                            space: "1d",
                            params: n + units,
                        });
                    }
                    _ => {}
                }
            }

            let inn = flat_len.unwrap_or(c * len);
            let id = format!("L{next_id}");
            layers.push(Layer::Output {
                inn,
                w: kaiming_uniform(inn, inn, rng),
                b: vec![0.0],
                gw: vec![0.0; inn],
                gb: vec![0.0],
                last_in: vec![],
                last_logit: 0.0,
            });
            metas.push(LayerMeta {
                id,
                kind: "output".into(),
                label: "Output".into(),
                channels: 1,
                rows: 0,
                cols: 0,
                length: 1,
                space: "1d",
                params: inn + 1,
            });
        }
    }

    (layers, metas)
}

pub fn forward_2d(layers: &mut [Layer], input: Volume) -> f32 {
    let mut cur_v = Some(input);
    let mut cur_s: Option<Signal> = None;
    let mut cur_f: Option<Vec<f32>> = None;
    let mut logit = 0.0;

    for layer in layers.iter_mut() {
        match layer {
            Layer::Input2d { .. } => {
                // keep cur_v
            }
            Layer::Conv2d {
                filters,
                k,
                in_c,
                act,
                w,
                b,
                last_in,
                last_pre,
                last_out,
                ..
            } => {
                let inp = cur_v.take().expect("conv2d needs volume");
                let (pre, out) =
                    crate::conv_gemm::conv2d_same_act(&inp, *filters, *k, *in_c, w, b, act);
                *last_in = inp;
                *last_pre = pre;
                *last_out = out.clone();
                cur_v = Some(out);
            }
            Layer::Pool2d {
                kind,
                last_in,
                last_out,
                switches,
            } => {
                let inp = cur_v.take().expect("pool needs volume");
                let c = inp.len();
                let h = inp[0].len();
                let w = inp[0][0].len();
                let oh = (h / 2).max(1);
                let ow = (w / 2).max(1);
                let mut out = zeros_volume(c, oh, ow);
                switches.clear();
                for ch in 0..c {
                    for y in 0..oh {
                        for x in 0..ow {
                            let y0 = y * 2;
                            let x0 = x * 2;
                            match kind {
                                PoolKind::Max => {
                                    let mut best = f32::NEG_INFINITY;
                                    let mut bi = 0usize;
                                    for dy in 0..2 {
                                        for dx in 0..2 {
                                            let yy = (y0 + dy).min(h - 1);
                                            let xx = (x0 + dx).min(w - 1);
                                            let v = inp[ch][yy][xx];
                                            if v > best {
                                                best = v;
                                                bi = yy * w + xx;
                                            }
                                        }
                                    }
                                    out[ch][y][x] = best;
                                    switches.push(bi);
                                }
                                PoolKind::Avg => {
                                    let mut s = 0.0;
                                    let mut n: f32 = 0.0;
                                    for dy in 0..2 {
                                        for dx in 0..2 {
                                            let yy = y0 + dy;
                                            let xx = x0 + dx;
                                            if yy < h && xx < w {
                                                s += inp[ch][yy][xx];
                                                n += 1.0;
                                            }
                                        }
                                    }
                                    out[ch][y][x] = s / n.max(1.0);
                                    switches.push(0);
                                }
                            }
                        }
                    }
                }
                *last_in = inp;
                *last_out = out.clone();
                cur_v = Some(out);
            }
            Layer::Gap2d { last_in, last_out } => {
                let inp = cur_v.take().expect("gap needs volume");
                let c = inp.len();
                let mut sig = zeros_signal(c, 1);
                for ch in 0..c {
                    let h = inp[ch].len();
                    let w = inp[ch][0].len();
                    let mut s = 0.0;
                    for row in &inp[ch] {
                        for &v in row {
                            s += v;
                        }
                    }
                    sig[ch][0] = s / (h * w) as f32;
                }
                *last_in = inp;
                *last_out = sig.clone();
                // flatten to vector for dense
                cur_f = Some(sig.iter().map(|r| r[0]).collect());
                cur_s = Some(sig);
                cur_v = None;
            }
            Layer::Flatten {
                c,
                h,
                w,
                last_flat,
            } => {
                let flat = if let Some(v) = cur_v.take() {
                    let mut f = Vec::with_capacity(*c * *h * *w);
                    for ch in 0..*c {
                        for y in 0..*h {
                            for x in 0..*w {
                                f.push(v[ch][y][x]);
                            }
                        }
                    }
                    f
                } else if let Some(s) = cur_s.take() {
                    s.into_iter().flatten().collect()
                } else {
                    cur_f.take().unwrap_or_default()
                };
                *last_flat = flat.clone();
                cur_f = Some(flat);
            }
            Layer::Dense {
                units,
                inn,
                act,
                w,
                b,
                last_in,
                last_pre,
                last_out,
                ..
            } => {
                let x = cur_f.take().expect("dense needs flat");
                let pre = dense_forward(&x, w, b, *units, *inn);
                let out: Vec<f32> = pre.iter().map(|&v| apply_act(act, v)).collect();
                *last_in = x;
                *last_pre = pre;
                *last_out = out.clone();
                cur_f = Some(out);
            }
            Layer::Output {
                inn,
                w,
                b,
                last_in,
                last_logit,
                ..
            } => {
                let x = cur_f.take().expect("output needs flat");
                let y = dense_forward(&x, w, b, 1, *inn);
                *last_in = x;
                *last_logit = y[0];
                logit = y[0];
            }
            _ => {}
        }
    }
    logit
}

pub fn forward_1d(layers: &mut [Layer], input: Signal) -> f32 {
    let mut cur_s = Some(input);
    let mut cur_f: Option<Vec<f32>> = None;
    let mut logit = 0.0;

    for layer in layers.iter_mut() {
        match layer {
            Layer::Input1d { .. } => {}
            Layer::Conv1d {
                filters,
                k,
                in_c,
                act,
                w,
                b,
                last_in,
                last_pre,
                last_out,
                ..
            } => {
                let inp = cur_s.take().expect("conv1d");
                let len = inp[0].len();
                let pad = *k / 2;
                let mut pre = zeros_signal(*filters, len);
                for o in 0..*filters {
                    for t in 0..len {
                        let mut s = b[o];
                        for ic in 0..*in_c {
                            for kk in 0..*k {
                                let it = t as isize + kk as isize - pad as isize;
                                if it >= 0 && (it as usize) < len {
                                    let wi = (o * *in_c + ic) * *k + kk;
                                    s += inp[ic][it as usize] * w[wi];
                                }
                            }
                        }
                        pre[o][t] = s;
                    }
                }
                let out: Signal = pre
                    .iter()
                    .map(|row| row.iter().map(|&v| apply_act(act, v)).collect())
                    .collect();
                *last_in = inp;
                *last_pre = pre;
                *last_out = out.clone();
                cur_s = Some(out);
            }
            Layer::Pool1d {
                kind,
                last_in,
                last_out,
                switches,
            } => {
                let inp = cur_s.take().unwrap();
                let c = inp.len();
                let len = inp[0].len();
                let ol = (len / 2).max(1);
                let mut out = zeros_signal(c, ol);
                switches.clear();
                for ch in 0..c {
                    for t in 0..ol {
                        let t0 = t * 2;
                        match kind {
                            PoolKind::Max => {
                                let mut best = f32::NEG_INFINITY;
                                let mut bi = 0;
                                for d in 0..2 {
                                    let tt = (t0 + d).min(len - 1);
                                    let v = inp[ch][tt];
                                    if v > best {
                                        best = v;
                                        bi = tt;
                                    }
                                }
                                out[ch][t] = best;
                                switches.push(bi);
                            }
                            PoolKind::Avg => {
                                let a = inp[ch][t0.min(len - 1)];
                                let b = inp[ch][(t0 + 1).min(len - 1)];
                                out[ch][t] = 0.5 * (a + b);
                                switches.push(0);
                            }
                        }
                    }
                }
                *last_in = inp;
                *last_out = out.clone();
                cur_s = Some(out);
            }
            Layer::Gap1d { last_in, last_out } => {
                let inp = cur_s.take().unwrap();
                let c = inp.len();
                let mut sig = zeros_signal(c, 1);
                for ch in 0..c {
                    let s: f32 = inp[ch].iter().sum::<f32>() / inp[ch].len().max(1) as f32;
                    sig[ch][0] = s;
                }
                *last_in = inp;
                *last_out = sig.clone();
                cur_f = Some(sig.iter().map(|r| r[0]).collect());
                cur_s = None;
            }
            Layer::Flatten { last_flat, .. } => {
                let flat = if let Some(s) = cur_s.take() {
                    s.into_iter().flatten().collect()
                } else {
                    cur_f.take().unwrap_or_default()
                };
                *last_flat = flat.clone();
                cur_f = Some(flat);
            }
            Layer::Dense {
                units,
                inn,
                act,
                w,
                b,
                last_in,
                last_pre,
                last_out,
                ..
            } => {
                let x = cur_f.take().unwrap();
                let pre = dense_forward(&x, w, b, *units, *inn);
                let out: Vec<f32> = pre.iter().map(|&v| apply_act(act, v)).collect();
                *last_in = x;
                *last_pre = pre;
                *last_out = out.clone();
                cur_f = Some(out);
            }
            Layer::Output {
                inn,
                w,
                b,
                last_in,
                last_logit,
                ..
            } => {
                let x = cur_f.take().unwrap();
                let y = dense_forward(&x, w, b, 1, *inn);
                *last_in = x;
                *last_logit = y[0];
                logit = y[0];
            }
            _ => {}
        }
    }
    logit
}

pub fn zero_grads(layers: &mut [Layer]) {
    for layer in layers {
        match layer {
            Layer::Conv2d { gw, gb, .. }
            | Layer::Conv1d { gw, gb, .. }
            | Layer::Dense { gw, gb, .. }
            | Layer::Output { gw, gb, .. } => {
                gw.fill(0.0);
                gb.fill(0.0);
            }
            _ => {}
        }
    }
}

/// BCE with logits: loss = softplus(logit) - label*logit; dlogit = sigmoid(logit)-label
pub fn bce_logit_grad(logit: f32, label: f32) -> (f32, f32) {
    let sig = 1.0 / (1.0 + (-logit).exp());
    let loss = if logit >= 0.0 {
        logit - label * logit + ((-logit).exp() + 1.0).ln()
    } else {
        -label * logit + (1.0 + logit.exp()).ln()
    };
    (loss, sig - label)
}

pub fn backward_from_logit(layers: &mut [Layer], dlogit: f32) {
    // walk reverse
    let mut grad_flat: Option<Vec<f32>> = None;
    let mut grad_sig: Option<Signal> = None;
    let mut grad_vol: Option<Volume> = None;

    for layer in layers.iter_mut().rev() {
        match layer {
            Layer::Output {
                inn,
                w,
                gw,
                gb,
                last_in,
                ..
            } => {
                let gy = vec![dlogit];
                let gx = dense_backward(last_in, w, &gy, 1, *inn, gw, gb);
                grad_flat = Some(gx);
            }
            Layer::Dense {
                units,
                inn,
                act,
                w,
                gw,
                gb,
                last_in,
                last_pre,
                last_out,
                ..
            } => {
                let mut gy = grad_flat.take().unwrap();
                for i in 0..*units {
                    gy[i] *= act_grad_from_out(act, last_out[i], last_pre[i]);
                }
                let gx = dense_backward(last_in, w, &gy, *units, *inn, gw, gb);
                grad_flat = Some(gx);
            }
            Layer::Gap2d { last_in, .. } => {
                let gflat = grad_flat.take().unwrap();
                let c = last_in.len();
                let h = last_in[0].len();
                let w = last_in[0][0].len();
                let scale = 1.0 / (h * w) as f32;
                let mut gv = zeros_volume(c, h, w);
                for ch in 0..c {
                    let g = gflat[ch] * scale;
                    for y in 0..h {
                        for x in 0..w {
                            gv[ch][y][x] = g;
                        }
                    }
                }
                grad_vol = Some(gv);
            }
            Layer::Gap1d { last_in, .. } => {
                let gflat = grad_flat.take().unwrap();
                let c = last_in.len();
                let len = last_in[0].len();
                let scale = 1.0 / len as f32;
                let mut gs = zeros_signal(c, len);
                for ch in 0..c {
                    let g = gflat[ch] * scale;
                    for t in 0..len {
                        gs[ch][t] = g;
                    }
                }
                grad_sig = Some(gs);
            }
            Layer::Flatten { c, h, w, .. } => {
                let gflat = grad_flat.take().unwrap();
                if *h > 0 && *w > 0 && gflat.len() == *c * *h * *w {
                    let mut gv = zeros_volume(*c, *h, *w);
                    let mut i = 0;
                    for ch in 0..*c {
                        for y in 0..*h {
                            for x in 0..*w {
                                gv[ch][y][x] = gflat[i];
                                i += 1;
                            }
                        }
                    }
                    grad_vol = Some(gv);
                } else {
                    // 1d flatten
                    grad_flat = Some(gflat);
                }
            }
            Layer::Pool2d {
                kind,
                last_in,
                last_out,
                switches,
            } => {
                let gout = grad_vol.take().unwrap();
                let c = last_in.len();
                let h = last_in[0].len();
                let w = last_in[0][0].len();
                let mut gin = zeros_volume(c, h, w);
                let mut si = 0;
                for ch in 0..c {
                    let oh = last_out[ch].len();
                    let ow = last_out[ch][0].len();
                    for y in 0..oh {
                        for x in 0..ow {
                            let g = gout[ch][y][x];
                            match kind {
                                PoolKind::Max => {
                                    let idx = switches[si];
                                    let yy = idx / w;
                                    let xx = idx % w;
                                    gin[ch][yy][xx] += g;
                                }
                                PoolKind::Avg => {
                                    let y0 = y * 2;
                                    let x0 = x * 2;
                                    let mut n: f32 = 0.0;
                                    for dy in 0..2 {
                                        for dx in 0..2 {
                                            if y0 + dy < h && x0 + dx < w {
                                                n += 1.0;
                                            }
                                        }
                                    }
                                    let share = g / n.max(1.0);
                                    for dy in 0..2 {
                                        for dx in 0..2 {
                                            if y0 + dy < h && x0 + dx < w {
                                                gin[ch][y0 + dy][x0 + dx] += share;
                                            }
                                        }
                                    }
                                }
                            }
                            si += 1;
                        }
                    }
                }
                grad_vol = Some(gin);
            }
            Layer::Conv2d {
                filters,
                k,
                in_c,
                act,
                w,
                gw,
                gb,
                last_in,
                last_pre,
                last_out,
                ..
            } => {
                let mut gout = grad_vol.take().unwrap();
                let h = last_out[0].len();
                let ww = last_out[0][0].len();
                for o in 0..*filters {
                    for y in 0..h {
                        for x in 0..ww {
                            gout[o][y][x] *=
                                act_grad_from_out(act, last_out[o][y][x], last_pre[o][y][x]);
                        }
                    }
                }
                let gin = crate::conv_gemm::conv2d_same_bwd(
                    &gout, last_in, *filters, *k, *in_c, w, gw, gb,
                );
                grad_vol = Some(gin);
            }
            Layer::Pool1d {
                kind,
                last_in,
                last_out,
                switches,
            } => {
                let gout = grad_sig.take().unwrap();
                let c = last_in.len();
                let len = last_in[0].len();
                let mut gin = zeros_signal(c, len);
                let mut si = 0;
                for ch in 0..c {
                    let ol = last_out[ch].len();
                    for t in 0..ol {
                        let g = gout[ch][t];
                        match kind {
                            PoolKind::Max => {
                                gin[ch][switches[si]] += g;
                            }
                            PoolKind::Avg => {
                                let t0 = t * 2;
                                gin[ch][t0.min(len - 1)] += 0.5 * g;
                                gin[ch][(t0 + 1).min(len - 1)] += 0.5 * g;
                            }
                        }
                        si += 1;
                    }
                }
                grad_sig = Some(gin);
            }
            Layer::Conv1d {
                filters,
                k,
                in_c,
                act,
                w,
                gw,
                gb,
                last_in,
                last_pre,
                last_out,
                ..
            } => {
                let mut gout = grad_sig.take().unwrap();
                let len = last_out[0].len();
                let pad = *k / 2;
                for o in 0..*filters {
                    for t in 0..len {
                        gout[o][t] *= act_grad_from_out(act, last_out[o][t], last_pre[o][t]);
                    }
                }
                let mut gin = zeros_signal(*in_c, last_in[0].len());
                for o in 0..*filters {
                    for t in 0..len {
                        let g = gout[o][t];
                        gb[o] += g;
                        for ic in 0..*in_c {
                            for kk in 0..*k {
                                let it = t as isize + kk as isize - pad as isize;
                                if it >= 0 && (it as usize) < last_in[0].len() {
                                    let wi = (o * *in_c + ic) * *k + kk;
                                    gw[wi] += g * last_in[ic][it as usize];
                                    gin[ic][it as usize] += g * w[wi];
                                }
                            }
                        }
                    }
                }
                grad_sig = Some(gin);
            }
            _ => {}
        }
    }
}

pub fn sgd_step(layers: &mut [Layer], lr: f32, batch_n: f32) {
    let scale = lr / batch_n.max(1.0);
    for layer in layers {
        match layer {
            Layer::Conv2d { w, b, gw, gb, .. }
            | Layer::Conv1d { w, b, gw, gb, .. }
            | Layer::Dense { w, b, gw, gb, .. }
            | Layer::Output { w, b, gw, gb, .. } => {
                for i in 0..w.len() {
                    w[i] -= scale * gw[i];
                }
                for i in 0..b.len() {
                    b[i] -= scale * gb[i];
                }
            }
            _ => {}
        }
    }
}

pub fn param_count(layers: &[Layer]) -> usize {
    let mut n = 0;
    for layer in layers {
        match layer {
            Layer::Conv2d { w, b, .. }
            | Layer::Conv1d { w, b, .. }
            | Layer::Dense { w, b, .. }
            | Layer::Output { w, b, .. } => n += w.len() + b.len(),
            _ => {}
        }
    }
    n
}

pub fn write_params(layers: &[Layer], dst: &mut [f32]) {
    let mut o = 0;
    for layer in layers {
        match layer {
            Layer::Conv2d { w, b, .. }
            | Layer::Conv1d { w, b, .. }
            | Layer::Dense { w, b, .. }
            | Layer::Output { w, b, .. } => {
                dst[o..o + w.len()].copy_from_slice(w);
                o += w.len();
                dst[o..o + b.len()].copy_from_slice(b);
                o += b.len();
            }
            _ => {}
        }
    }
}

pub fn read_params(layers: &mut [Layer], src: &[f32]) {
    let mut o = 0;
    for layer in layers {
        match layer {
            Layer::Conv2d { w, b, .. }
            | Layer::Conv1d { w, b, .. }
            | Layer::Dense { w, b, .. }
            | Layer::Output { w, b, .. } => {
                let wn = w.len();
                let bn = b.len();
                w.copy_from_slice(&src[o..o + wn]);
                o += wn;
                b.copy_from_slice(&src[o..o + bn]);
                o += bn;
            }
            _ => {}
        }
    }
}

pub fn write_grads(layers: &[Layer], dst: &mut [f32]) {
    let mut o = 0;
    for layer in layers {
        match layer {
            Layer::Conv2d { gw, gb, .. }
            | Layer::Conv1d { gw, gb, .. }
            | Layer::Dense { gw, gb, .. }
            | Layer::Output { gw, gb, .. } => {
                dst[o..o + gw.len()].copy_from_slice(gw);
                o += gw.len();
                dst[o..o + gb.len()].copy_from_slice(gb);
                o += gb.len();
            }
            _ => {}
        }
    }
}

pub fn add_grads(layers: &mut [Layer], src: &[f32]) {
    let mut o = 0;
    for layer in layers {
        match layer {
            Layer::Conv2d { gw, gb, .. }
            | Layer::Conv1d { gw, gb, .. }
            | Layer::Dense { gw, gb, .. }
            | Layer::Output { gw, gb, .. } => {
                for i in 0..gw.len() {
                    gw[i] += src[o + i];
                }
                o += gw.len();
                for i in 0..gb.len() {
                    gb[i] += src[o + i];
                }
                o += gb.len();
            }
            _ => {}
        }
    }
}

pub fn display_kernels_2d(layer: &Layer) -> Option<(Vec<Map2D>, Vec<f32>)> {
    let Layer::Conv2d {
        filters,
        k,
        in_c,
        w,
        b,
        ..
    } = layer
    else {
        return None;
    };
    let mut kernels = Vec::with_capacity(*filters);
    for o in 0..*filters {
        let mut map = vec![vec![0.0; *k]; *k];
        for ic in 0..*in_c {
            for ky in 0..*k {
                for kx in 0..*k {
                    let wi = ((o * *in_c + ic) * *k + ky) * *k + kx;
                    map[ky][kx] += w[wi];
                }
            }
        }
        kernels.push(map);
    }
    Some((kernels, b.clone()))
}

/// Per-input-channel kernels: `[out][in][ky][kx]` (no summing).
pub fn display_kernels_2d_in(layer: &Layer) -> Option<Vec<Vec<Map2D>>> {
    let Layer::Conv2d {
        filters,
        k,
        in_c,
        w,
        ..
    } = layer
    else {
        return None;
    };
    let mut out = Vec::with_capacity(*filters);
    for o in 0..*filters {
        let mut per_in = Vec::with_capacity(*in_c);
        for ic in 0..*in_c {
            let mut map = vec![vec![0.0; *k]; *k];
            for ky in 0..*k {
                for kx in 0..*k {
                    let wi = ((o * *in_c + ic) * *k + ky) * *k + kx;
                    map[ky][kx] = w[wi];
                }
            }
            per_in.push(map);
        }
        out.push(per_in);
    }
    Some(out)
}

pub fn display_kernels_1d(layer: &Layer) -> Option<(Vec<Vec<f32>>, Vec<f32>)> {
    let Layer::Conv1d {
        filters,
        k,
        in_c,
        w,
        b,
        ..
    } = layer
    else {
        return None;
    };
    let mut kernels = Vec::with_capacity(*filters);
    for o in 0..*filters {
        let mut row = vec![0.0; *k];
        for ic in 0..*in_c {
            for kk in 0..*k {
                let wi = (o * *in_c + ic) * *k + kk;
                row[kk] += w[wi];
            }
        }
        kernels.push(row);
    }
    Some((kernels, b.clone()))
}

/// Per-input-channel 1-D kernels: `[out][in][k]`.
pub fn display_kernels_1d_in(layer: &Layer) -> Option<Vec<Vec<Vec<f32>>>> {
    let Layer::Conv1d {
        filters,
        k,
        in_c,
        w,
        ..
    } = layer
    else {
        return None;
    };
    let mut out = Vec::with_capacity(*filters);
    for o in 0..*filters {
        let mut per_in = Vec::with_capacity(*in_c);
        for ic in 0..*in_c {
            let mut row = vec![0.0; *k];
            for kk in 0..*k {
                let wi = (o * *in_c + ic) * *k + kk;
                row[kk] = w[wi];
            }
            per_in.push(row);
        }
        out.push(per_in);
    }
    Some(out)
}

pub fn dense_matrix(layer: &Layer) -> Option<Vec<Vec<f32>>> {
    match layer {
        Layer::Dense { units, inn, w, .. } => {
            let mut m = vec![vec![0.0; *inn]; *units];
            for o in 0..*units {
                for i in 0..*inn {
                    m[o][i] = w[o * *inn + i];
                }
            }
            Some(m)
        }
        Layer::Output { inn, w, .. } => {
            let mut m = vec![vec![0.0; *inn]; 1];
            for i in 0..*inn {
                m[0][i] = w[i];
            }
            Some(m)
        }
        _ => None,
    }
}

pub fn weight_mag(layer: &Layer) -> Option<f64> {
    let w = match layer {
        Layer::Conv2d { w, .. }
        | Layer::Conv1d { w, .. }
        | Layer::Dense { w, .. }
        | Layer::Output { w, .. } => w,
        _ => return None,
    };
    if w.is_empty() {
        return Some(0.0);
    }
    let s: f64 = w.iter().map(|v| v.abs() as f64).sum();
    Some((s / w.len() as f64).tanh())
}
