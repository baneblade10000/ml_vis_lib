//! Tiny tensor helpers (NCHW / flat vectors). No Burn.

use rand::Rng;

pub type Map2D = Vec<Vec<f32>>;
pub type Volume = Vec<Map2D>; // [C][H][W]
pub type Signal = Vec<Vec<f32>>; // [C][L]

pub fn zeros2d(h: usize, w: usize) -> Map2D {
    vec![vec![0.0; w]; h]
}

pub fn zeros_volume(c: usize, h: usize, w: usize) -> Volume {
    (0..c).map(|_| zeros2d(h, w)).collect()
}

pub fn zeros_signal(c: usize, len: usize) -> Signal {
    vec![vec![0.0; len]; c]
}

pub fn kaiming_uniform(fan_in: usize, n: usize, rng: &mut impl Rng) -> Vec<f32> {
    let bound = (6.0 / fan_in.max(1) as f32).sqrt();
    (0..n).map(|_| rng.gen_range(-bound..bound)).collect()
}

#[inline]
pub fn relu(x: f32) -> f32 {
    x.max(0.0)
}
#[inline]
pub fn relu_grad(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else {
        0.0
    }
}
#[inline]
pub fn tanh_a(x: f32) -> f32 {
    x.tanh()
}
#[inline]
pub fn tanh_grad(y: f32) -> f32 {
    1.0 - y * y
}
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
#[inline]
pub fn sigmoid_grad(y: f32) -> f32 {
    y * (1.0 - y)
}

pub fn apply_act(id: &str, x: f32) -> f32 {
    match id {
        "tanh" => tanh_a(x),
        "sigmoid" => sigmoid(x),
        "linear" => x,
        _ => relu(x),
    }
}

pub fn act_grad_from_out(id: &str, y: f32, pre: f32) -> f32 {
    match id {
        "tanh" => tanh_grad(y),
        "sigmoid" => sigmoid_grad(y),
        "linear" => 1.0,
        _ => relu_grad(pre),
    }
}

/// Dense: y[o] = sum_i x[i]*W[o*in+i] + b[o]  (row-major W[out][in])
pub fn dense_forward(x: &[f32], w: &[f32], b: &[f32], out: usize, inn: usize) -> Vec<f32> {
    let mut y = vec![0.0; out];
    for o in 0..out {
        let mut s = b[o];
        let row = o * inn;
        for i in 0..inn {
            s += x[i] * w[row + i];
        }
        y[o] = s;
    }
    y
}

pub fn dense_backward(
    x: &[f32],
    w: &[f32],
    grad_y: &[f32],
    out: usize,
    inn: usize,
    grad_w: &mut [f32],
    grad_b: &mut [f32],
) -> Vec<f32> {
    let mut grad_x = vec![0.0; inn];
    for o in 0..out {
        let gy = grad_y[o];
        grad_b[o] += gy;
        let row = o * inn;
        for i in 0..inn {
            grad_w[row + i] += gy * x[i];
            grad_x[i] += gy * w[row + i];
        }
    }
    grad_x
}
