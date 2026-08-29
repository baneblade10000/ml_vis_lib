//! im2col + SIMD GEMM for same-padding conv2d.
//!
//! Layouts (row-major unless noted):
//! - `col`: K×N with `col[ki * n + ni]` (patch dim major) — contiguous N for f32x4
//! - `w`:   M×K with `w[o * k + ki]`
//! - `y`:   M×N with `y[o * n + ni]`

use crate::simd_f32::F32x4;
use crate::tensor::{apply_act, zeros_volume, Volume};

#[inline]
pub fn pad_chw(input: &Volume, pad: usize) -> (Vec<f32>, usize, usize, usize, usize, usize) {
    let in_c = input.len();
    let h = input[0].len();
    let w = input[0][0].len();
    let ph = h + 2 * pad;
    let pw = w + 2 * pad;
    let mut pinned = vec![0.0f32; in_c * ph * pw];
    for ic in 0..in_c {
        let base = ic * ph * pw;
        for y in 0..h {
            let dst = base + (y + pad) * pw + pad;
            pinned[dst..dst + w].copy_from_slice(&input[ic][y]);
        }
    }
    (pinned, in_c, h, w, ph, pw)
}

/// Unfold padded CHW → K×N (`K = in_c*k*k`, `N = h*w`).
pub fn im2col(pinned: &[f32], in_c: usize, h: usize, w: usize, ph: usize, pw: usize, k: usize) -> Vec<f32> {
    let n = h * w;
    let kk = k * k;
    let kd = in_c * kk;
    let mut col = vec![0.0f32; kd * n];
    for y in 0..h {
        for x in 0..w {
            let ni = y * w + x;
            for ic in 0..in_c {
                let src_base = ic * ph * pw + y * pw + x;
                let ki0 = ic * kk;
                for ky in 0..k {
                    let prow = src_base + ky * pw;
                    let ki_row = ki0 + ky * k;
                    for kx in 0..k {
                        col[(ki_row + kx) * n + ni] = pinned[prow + kx];
                    }
                }
            }
        }
    }
    col
}

/// Scatter K×N grads back into padded CHW (accumulate).
pub fn col2im(
    dcol: &[f32],
    in_c: usize,
    h: usize,
    w: usize,
    ph: usize,
    pw: usize,
    k: usize,
    gin_pad: &mut [f32],
) {
    let n = h * w;
    let kk = k * k;
    for y in 0..h {
        for x in 0..w {
            let ni = y * w + x;
            for ic in 0..in_c {
                let dst_base = ic * ph * pw + y * pw + x;
                let ki0 = ic * kk;
                for ky in 0..k {
                    let prow = dst_base + ky * pw;
                    let ki_row = ki0 + ky * k;
                    for kx in 0..k {
                        gin_pad[prow + kx] += dcol[(ki_row + kx) * n + ni];
                    }
                }
            }
        }
    }
}

/// Y[M,N] = bias + W[M,K] · Col[K,N]  (vectorized along N).
pub fn gemm_w_col(m: usize, n: usize, k: usize, w: &[f32], col: &[f32], bias: &[f32], y: &mut [f32]) {
    debug_assert_eq!(w.len(), m * k);
    debug_assert_eq!(col.len(), k * n);
    debug_assert_eq!(y.len(), m * n);
    for o in 0..m {
        let w_row = o * k;
        let y_row = o * n;
        let mut ni = 0usize;
        while ni + 4 <= n {
            let mut acc = F32x4::splat(bias[o]);
            for ki in 0..k {
                let a = F32x4::splat(w[w_row + ki]);
                let b = unsafe { F32x4::load(col.as_ptr().add(ki * n + ni)) };
                acc = acc.mul_add(a, b);
            }
            unsafe {
                acc.store(y.as_mut_ptr().add(y_row + ni));
            }
            ni += 4;
        }
        while ni < n {
            let mut s = bias[o];
            for ki in 0..k {
                s += w[w_row + ki] * col[ki * n + ni];
            }
            y[y_row + ni] = s;
            ni += 1;
        }
    }
}

/// dW[M,K] += G[M,N] · Colᵀ[N,K]  →  dW[o,ki] += Σ_n G[o,n] * Col[ki,n]
pub fn gemm_gw_outer(m: usize, n: usize, k: usize, g: &[f32], col: &[f32], gw: &mut [f32]) {
    for o in 0..m {
        let g_row = o * n;
        let w_row = o * k;
        for ki in 0..k {
            let mut acc = F32x4::splat(0.0);
            let mut ni = 0usize;
            let col_row = ki * n;
            while ni + 4 <= n {
                let gv = unsafe { F32x4::load(g.as_ptr().add(g_row + ni)) };
                let cv = unsafe { F32x4::load(col.as_ptr().add(col_row + ni)) };
                acc = acc.mul_add(gv, cv);
                ni += 4;
            }
            let mut s = acc.sum();
            while ni < n {
                s += g[g_row + ni] * col[col_row + ni];
                ni += 1;
            }
            gw[w_row + ki] += s;
        }
    }
}

/// dCol[K,N] = Wᵀ[K,M] · G[M,N]  →  dCol[ki,n] = Σ_o W[o,ki] * G[o,n]
pub fn gemm_dcol(m: usize, n: usize, k: usize, w: &[f32], g: &[f32], dcol: &mut [f32]) {
    dcol.fill(0.0);
    for o in 0..m {
        let w_row = o * k;
        let g_row = o * n;
        for ki in 0..k {
            let wv = F32x4::splat(w[w_row + ki]);
            let col_row = ki * n;
            let mut ni = 0usize;
            while ni + 4 <= n {
                let gv = unsafe { F32x4::load(g.as_ptr().add(g_row + ni)) };
                let dst = unsafe { dcol.as_mut_ptr().add(col_row + ni) };
                let acc = F32x4::load(dst).mul_add(wv, gv);
                acc.store(dst);
                ni += 4;
            }
            while ni < n {
                dcol[col_row + ni] += w[w_row + ki] * g[g_row + ni];
                ni += 1;
            }
        }
    }
}

fn flat_to_volume(flat: &[f32], m: usize, h: usize, w: usize) -> Volume {
    let n = h * w;
    let mut vol = zeros_volume(m, h, w);
    for o in 0..m {
        for y in 0..h {
            let src = o * n + y * w;
            vol[o][y].copy_from_slice(&flat[src..src + w]);
        }
    }
    vol
}

fn volume_to_flat(vol: &Volume) -> (Vec<f32>, usize, usize, usize) {
    let m = vol.len();
    let h = vol[0].len();
    let w = vol[0][0].len();
    let n = h * w;
    let mut flat = vec![0.0f32; m * n];
    for o in 0..m {
        for y in 0..h {
            let dst = o * n + y * w;
            flat[dst..dst + w].copy_from_slice(&vol[o][y]);
        }
    }
    (flat, m, h, w)
}

/// Forward: same-pad conv2d + activation via im2col + GEMM.
pub fn conv2d_same_act(
    input: &Volume,
    filters: usize,
    k: usize,
    in_c: usize,
    weights: &[f32],
    bias: &[f32],
    act: &str,
) -> (Volume, Volume) {
    let pad = k / 2;
    let (pinned, ic, h, w, ph, pw) = pad_chw(input, pad);
    debug_assert_eq!(ic, in_c);
    let n = h * w;
    let kd = in_c * k * k;
    let col = im2col(&pinned, in_c, h, w, ph, pw, k);
    let mut pre_flat = vec![0.0f32; filters * n];
    gemm_w_col(filters, n, kd, weights, &col, bias, &mut pre_flat);
    let pre = flat_to_volume(&pre_flat, filters, h, w);
    let mut out = zeros_volume(filters, h, w);
    for o in 0..filters {
        for y in 0..h {
            for x in 0..w {
                out[o][y][x] = apply_act(act, pre[o][y][x]);
            }
        }
    }
    (pre, out)
}

/// Backward grads for conv2d (after activation grad already applied to `gout`).
/// Accumulates into `gw` / `gb`; returns `gin` volume.
pub fn conv2d_same_bwd(
    gout: &Volume,
    last_in: &Volume,
    filters: usize,
    k: usize,
    in_c: usize,
    weights: &[f32],
    gw: &mut [f32],
    gb: &mut [f32],
) -> Volume {
    let pad = k / 2;
    let (pinned, ic, ih, iw, ph, pw) = pad_chw(last_in, pad);
    debug_assert_eq!(ic, in_c);
    let h = gout[0].len();
    let w = gout[0][0].len();
    debug_assert_eq!(h, ih);
    debug_assert_eq!(w, iw);
    let n = h * w;
    let kd = in_c * k * k;
    let col = im2col(&pinned, in_c, h, w, ph, pw, k);
    let (g_flat, m, _, _) = volume_to_flat(gout);
    debug_assert_eq!(m, filters);

    for o in 0..filters {
        let base = o * n;
        let mut s = 0.0f32;
        for ni in 0..n {
            s += g_flat[base + ni];
        }
        gb[o] += s;
    }

    gemm_gw_outer(filters, n, kd, &g_flat, &col, gw);

    let mut dcol = vec![0.0f32; kd * n];
    gemm_dcol(filters, n, kd, weights, &g_flat, &mut dcol);

    let mut gin_pad = vec![0.0f32; in_c * ph * pw];
    col2im(&dcol, in_c, h, w, ph, pw, k, &mut gin_pad);

    let mut gin = zeros_volume(in_c, ih, iw);
    for ic in 0..in_c {
        let base = ic * ph * pw;
        for y in 0..ih {
            let src = base + (y + pad) * pw + pad;
            gin[ic][y].copy_from_slice(&gin_pad[src..src + iw]);
        }
    }
    gin
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::zeros_volume;

    fn naive_conv(input: &Volume, filters: usize, k: usize, in_c: usize, w: &[f32], b: &[f32]) -> Volume {
        let h = input[0].len();
        let ww = input[0][0].len();
        let pad = k / 2;
        let mut out = zeros_volume(filters, h, ww);
        for o in 0..filters {
            for y in 0..h {
                for x in 0..ww {
                    let mut s = b[o];
                    for ic in 0..in_c {
                        for ky in 0..k {
                            for kx in 0..k {
                                let iy = y as isize + ky as isize - pad as isize;
                                let ix = x as isize + kx as isize - pad as isize;
                                if iy >= 0 && ix >= 0 && (iy as usize) < h && (ix as usize) < ww {
                                    let wi = ((o * in_c + ic) * k + ky) * k + kx;
                                    s += input[ic][iy as usize][ix as usize] * w[wi];
                                }
                            }
                        }
                    }
                    out[o][y][x] = s;
                }
            }
        }
        out
    }

    #[test]
    fn im2col_gemm_matches_naive() {
        let h = 8usize;
        let w = 8usize;
        let in_c = 3usize;
        let filters = 4usize;
        let k = 3usize;
        let mut input = zeros_volume(in_c, h, w);
        for ic in 0..in_c {
            for y in 0..h {
                for x in 0..w {
                    input[ic][y][x] = ((ic * 17 + y * 3 + x) % 11) as f32 * 0.1;
                }
            }
        }
        let kd = in_c * k * k;
        let weights: Vec<f32> = (0..filters * kd).map(|i| ((i * 13) % 7) as f32 * 0.05 - 0.1).collect();
        let bias: Vec<f32> = (0..filters).map(|i| i as f32 * 0.01).collect();
        let (pre, _) = conv2d_same_act(&input, filters, k, in_c, &weights, &bias, "linear");
        let naive = naive_conv(&input, filters, k, in_c, &weights, &bias);
        for o in 0..filters {
            for y in 0..h {
                for x in 0..w {
                    let a = pre[o][y][x];
                    let b = naive[o][y][x];
                    assert!((a - b).abs() < 1e-4, "o={o} y={y} x={x}: {a} vs {b}");
                }
            }
        }
    }
}
