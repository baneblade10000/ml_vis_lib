//! Row-major f32 matrix plus the handful of GEMM shapes the transformer needs.

#[derive(Clone, Debug)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Mat {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn from_fn(rows: usize, cols: usize, mut f: impl FnMut(usize, usize) -> f32) -> Self {
        let mut data = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                data.push(f(r, c));
            }
        }
        Self { rows, cols, data }
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> f32 {
        self.data[r * self.cols + c]
    }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, v: f32) {
        self.data[r * self.cols + c] = v;
    }

    pub fn add_assign(&mut self, other: &Mat) {
        debug_assert_eq!(self.rows, other.rows);
        debug_assert_eq!(self.cols, other.cols);
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a += b;
        }
    }

    pub fn zero(&mut self) {
        self.data.iter_mut().for_each(|v| *v = 0.0);
    }

    /// Row `r` as a slice — used by row-wise softmax / layernorm.
    #[inline]
    pub fn row(&self, r: usize) -> &[f32] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }

    #[inline]
    pub fn row_mut(&mut self, r: usize) -> &mut [f32] {
        let c = self.cols;
        &mut self.data[r * c..(r + 1) * c]
    }
}

/// `a @ b`
pub fn matmul(a: &Mat, b: &Mat) -> Mat {
    debug_assert_eq!(a.cols, b.rows);
    let mut out = Mat::zeros(a.rows, b.cols);
    for i in 0..a.rows {
        let a_row = &a.data[i * a.cols..(i + 1) * a.cols];
        let o_row = &mut out.data[i * b.cols..(i + 1) * b.cols];
        for (k, &av) in a_row.iter().enumerate() {
            if av == 0.0 {
                continue;
            }
            let b_row = &b.data[k * b.cols..(k + 1) * b.cols];
            for (o, &bv) in o_row.iter_mut().zip(b_row.iter()) {
                *o += av * bv;
            }
        }
    }
    out
}

/// `a @ bᵀ` — the attention score shape (`queries × keys`) without a transpose copy.
pub fn matmul_at_b(a: &Mat, b: &Mat) -> Mat {
    debug_assert_eq!(a.cols, b.cols);
    let mut out = Mat::zeros(a.rows, b.rows);
    for i in 0..a.rows {
        let a_row = a.row(i);
        for j in 0..b.rows {
            let b_row = b.row(j);
            let mut s = 0.0f32;
            for (x, y) in a_row.iter().zip(b_row.iter()) {
                s += x * y;
            }
            out.set(i, j, s);
        }
    }
    out
}

/// `aᵀ @ b` — the weight-gradient shape (`in_features × out_features`).
pub fn matmul_t_a(a: &Mat, b: &Mat) -> Mat {
    debug_assert_eq!(a.rows, b.rows);
    let mut out = Mat::zeros(a.cols, b.cols);
    for i in 0..a.rows {
        let a_row = a.row(i);
        let b_row = b.row(i);
        for (k, &av) in a_row.iter().enumerate() {
            if av == 0.0 {
                continue;
            }
            for (j, &bv) in b_row.iter().enumerate() {
                out.data[k * b.cols + j] += av * bv;
            }
        }
    }
    out
}
