//! Portable f32x4 helpers for conv hot paths.
//!
//! Nested `Vec<Vec<…>>` loops don't auto-vectorize. Flat rows + explicit SIMD do.
//! WASM needs `+simd128` (see `.cargo/config.toml`); native uses SSE2 (always on x86_64).

#![allow(unsafe_code)]

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use core::arch::wasm32::{
    f32x4_add, f32x4_extract_lane, f32x4_mul, f32x4_splat, v128, v128_load, v128_store,
};

#[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
use core::arch::x86_64::{
    __m128, _mm_add_ps, _mm_loadu_ps, _mm_mul_ps, _mm_set1_ps, _mm_storeu_ps,
};

#[derive(Copy, Clone)]
pub struct F32x4(
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))] v128,
    #[cfg(all(target_arch = "x86_64", target_feature = "sse"))] __m128,
    #[cfg(not(any(
        all(target_arch = "wasm32", target_feature = "simd128"),
        all(target_arch = "x86_64", target_feature = "sse"),
    )))]
    [f32; 4],
);

impl F32x4 {
    #[inline(always)]
    pub fn splat(v: f32) -> Self {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            Self(f32x4_splat(v))
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
        {
            Self(unsafe { _mm_set1_ps(v) })
        }
        #[cfg(not(any(
            all(target_arch = "wasm32", target_feature = "simd128"),
            all(target_arch = "x86_64", target_feature = "sse"),
        )))]
        {
            Self([v, v, v, v])
        }
    }

    #[inline(always)]
    pub fn load(ptr: *const f32) -> Self {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            Self(unsafe { v128_load(ptr as *const v128) })
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
        {
            Self(unsafe { _mm_loadu_ps(ptr) })
        }
        #[cfg(not(any(
            all(target_arch = "wasm32", target_feature = "simd128"),
            all(target_arch = "x86_64", target_feature = "sse"),
        )))]
        {
            unsafe { Self([*ptr, *ptr.add(1), *ptr.add(2), *ptr.add(3)]) }
        }
    }

    #[inline(always)]
    pub fn store(self, ptr: *mut f32) {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        unsafe {
            v128_store(ptr as *mut v128, self.0);
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
        unsafe {
            _mm_storeu_ps(ptr, self.0);
        }
        #[cfg(not(any(
            all(target_arch = "wasm32", target_feature = "simd128"),
            all(target_arch = "x86_64", target_feature = "sse"),
        )))]
        unsafe {
            *ptr = self.0[0];
            *ptr.add(1) = self.0[1];
            *ptr.add(2) = self.0[2];
            *ptr.add(3) = self.0[3];
        }
    }

    #[inline(always)]
    pub fn mul_add(self, a: Self, b: Self) -> Self {
        // acc + a * b
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            Self(f32x4_add(self.0, f32x4_mul(a.0, b.0)))
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
        {
            Self(unsafe { _mm_add_ps(self.0, _mm_mul_ps(a.0, b.0)) })
        }
        #[cfg(not(any(
            all(target_arch = "wasm32", target_feature = "simd128"),
            all(target_arch = "x86_64", target_feature = "sse"),
        )))]
        {
            Self([
                self.0[0] + a.0[0] * b.0[0],
                self.0[1] + a.0[1] * b.0[1],
                self.0[2] + a.0[2] * b.0[2],
                self.0[3] + a.0[3] * b.0[3],
            ])
        }
    }

    #[inline(always)]
    pub fn add(self, other: Self) -> Self {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            Self(f32x4_add(self.0, other.0))
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
        {
            Self(unsafe { _mm_add_ps(self.0, other.0) })
        }
        #[cfg(not(any(
            all(target_arch = "wasm32", target_feature = "simd128"),
            all(target_arch = "x86_64", target_feature = "sse"),
        )))]
        {
            Self([
                self.0[0] + other.0[0],
                self.0[1] + other.0[1],
                self.0[2] + other.0[2],
                self.0[3] + other.0[3],
            ])
        }
    }

    #[inline(always)]
    pub fn mul(self, other: Self) -> Self {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            Self(f32x4_mul(self.0, other.0))
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
        {
            Self(unsafe { _mm_mul_ps(self.0, other.0) })
        }
        #[cfg(not(any(
            all(target_arch = "wasm32", target_feature = "simd128"),
            all(target_arch = "x86_64", target_feature = "sse"),
        )))]
        {
            Self([
                self.0[0] * other.0[0],
                self.0[1] * other.0[1],
                self.0[2] * other.0[2],
                self.0[3] * other.0[3],
            ])
        }
    }

    #[inline(always)]
    pub fn sum(self) -> f32 {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            f32x4_extract_lane::<0>(self.0)
                + f32x4_extract_lane::<1>(self.0)
                + f32x4_extract_lane::<2>(self.0)
                + f32x4_extract_lane::<3>(self.0)
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
        {
            let mut tmp = [0.0f32; 4];
            unsafe { _mm_storeu_ps(tmp.as_mut_ptr(), self.0) };
            tmp[0] + tmp[1] + tmp[2] + tmp[3]
        }
        #[cfg(not(any(
            all(target_arch = "wasm32", target_feature = "simd128"),
            all(target_arch = "x86_64", target_feature = "sse"),
        )))]
        {
            self.0[0] + self.0[1] + self.0[2] + self.0[3]
        }
    }
}
