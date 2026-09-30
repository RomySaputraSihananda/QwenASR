//! Generic (portable) implementations of hot kernels.

#[inline]
pub fn bf16_to_f32(bf16: u16) -> f32 {
    f32::from_bits((bf16 as u32) << 16)
}

/// # Safety
/// w_bf16 must point to at least out_dim * in_dim valid bf16 values.
pub unsafe fn bf16_matvec_fused(
    y: &mut [f32],
    x: &[f32],
    w_bf16: *const u16,
    bias: Option<&[f32]>,
    in_dim: usize,
    out_dim: usize,
) {
    for o in 0..out_dim {
        let w_row = unsafe { std::slice::from_raw_parts(w_bf16.add(o * in_dim), in_dim) };
        let mut sum = bias.map_or(0.0f32, |b| b[o]);
        for k in 0..in_dim {
            sum += bf16_to_f32(w_row[k]) * x[k];
        }
        y[o] = sum;
    }
}

/// # Safety
/// w_bf16 must point to at least end * in_dim valid bf16 values.
pub unsafe fn argmax_bf16_range(
    x: &[f32],
    w_bf16: *const u16,
    in_dim: usize,
    start: usize,
    end: usize,
) -> (usize, f32) {
    let mut best = start;
    let mut best_val = -1e30f32;

    for o in start..end {
        let w_row = unsafe { std::slice::from_raw_parts(w_bf16.add(o * in_dim), in_dim) };
        let mut sum = 0.0f32;
        for k in 0..in_dim {
            sum += bf16_to_f32(w_row[k]) * x[k];
        }
        if sum > best_val {
            best_val = sum;
            best = o;
        }
    }
    (best, best_val)
}

pub fn dot_f32(a: &[f32], b: &[f32], n: usize) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..n {
        sum += a[i] * b[i];
    }
    sum
}

pub fn vec_scale_inplace(dst: &mut [f32], scale: f32, n: usize) {
    for val in dst.iter_mut().take(n) {
        *val *= scale;
    }
}

pub fn vec_axpy_inplace(dst: &mut [f32], src: &[f32], alpha: f32, n: usize) {
    for i in 0..n {
        dst[i] += alpha * src[i];
    }
}

pub fn vec_scale_add(dst: &mut [f32], src: &[f32], correction: f32, n: usize) {
    for i in 0..n {
        dst[i] = dst[i] * correction + src[i];
    }
}

/// Convert a buffer of BF16 values (as raw u16) to f32, element-wise.
/// Portable equivalent of `avx::bf16_to_f32_buf` / `neon`'s buffer converters.
pub fn bf16_to_f32_buf(dst: &mut [f32], src: &[u16]) {
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d = bf16_to_f32(s);
    }
}

/// RMS norm for a single row: `out = x / rms(x) * weight`, where
/// `rms(x) = sqrt(mean(x^2) + eps)`. Portable equivalent of `avx::rms_norm_row`.
pub fn rms_norm_row(out: &mut [f32], x: &[f32], weight: &[f32], hidden: usize, eps: f32) {
    let mut sum_sq = 0.0f32;
    for &v in x.iter().take(hidden) {
        sum_sq += v * v;
    }
    let rms_inv = 1.0 / (sum_sq / hidden as f32 + eps).sqrt();
    for i in 0..hidden {
        out[i] = x[i] * rms_inv * weight[i];
    }
}

/// Layer norm for a single row: `out = (x - mean(x)) / sqrt(var(x) + eps) *
/// weight + bias`. Portable equivalent of `avx::layer_norm_row`.
#[allow(clippy::too_many_arguments)]
pub fn layer_norm_row(
    out: &mut [f32],
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    hidden: usize,
    eps: f32,
) {
    let mut mean = 0.0f32;
    for &v in x.iter().take(hidden) {
        mean += v;
    }
    mean /= hidden as f32;

    let mut var = 0.0f32;
    for &v in x.iter().take(hidden) {
        let d = v - mean;
        var += d * d;
    }
    let inv_std = 1.0 / (var / hidden as f32 + eps).sqrt();

    for i in 0..hidden {
        out[i] = (x[i] - mean) * inv_std * weight[i] + bias[i];
    }
}

/// Exact `exp()` (via libm through `f32::exp`), element-wise in place.
/// Portable equivalent of `avx::exp_inplace` -- that one trades accuracy
/// (~1e-4 relative error) for speed via a polynomial approximation; this one
/// doesn't need to, so it just uses the real thing.
pub fn exp_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = v.exp();
    }
}

/// GELU (tanh approximation), element-wise in place. Portable equivalent of
/// `avx::gelu_inplace` -- same formula as that function's own scalar tail
/// (`0.5 * val * (1 + tanh(sqrt(2/pi) * (val + 0.044715 * val^3)))`), just
/// applied to the whole buffer instead of only the non-vectorizable remainder.
pub fn gelu_inplace(x: &mut [f32], n: usize) {
    const COEFF: f32 = 0.797_884_6; // sqrt(2/pi), same truncation avx.rs uses
    const C3: f32 = 0.044715;
    for val in x.iter_mut().take(n) {
        let v = *val;
        let x3 = v * v * v;
        let inner = COEFF * (v + C3 * x3);
        *val = 0.5 * v * (1.0 + inner.tanh());
    }
}

/// Quantize a BF16 weight matrix to INT8 per-row with absmax scaling.
/// Returns `(int8_data, scales)` where `scales[row]` is that row's scale
/// factor (`int8_data[row][k] ≈ w[row][k] / scales[row]`). Portable
/// equivalent of `avx::quantize_bf16_to_int8` -- same per-row absmax scan and
/// the same round-and-clamp formula that function's own scalar tail uses,
/// just applied to every element instead of only the remainder.
///
/// # Safety
/// `w_bf16` must point to at least `out_dim * in_dim` valid BF16 values.
pub unsafe fn quantize_bf16_to_int8(
    w_bf16: *const u16,
    out_dim: usize,
    in_dim: usize,
) -> (Vec<i8>, Vec<f32>) {
    let mut int8_data = vec![0i8; out_dim * in_dim];
    let mut scales = vec![0.0f32; out_dim];

    for row in 0..out_dim {
        let w_row = unsafe { std::slice::from_raw_parts(w_bf16.add(row * in_dim), in_dim) };

        let mut max_abs = 0.0f32;
        for &w in w_row {
            let v = bf16_to_f32(w).abs();
            if v > max_abs {
                max_abs = v;
            }
        }

        let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
        let inv_scale = 127.0 / max_abs.max(1e-10);
        scales[row] = scale;

        let dst = &mut int8_data[row * in_dim..(row + 1) * in_dim];
        for k in 0..in_dim {
            let v = bf16_to_f32(w_row[k]);
            dst[k] = (v * inv_scale).round().clamp(-127.0, 127.0) as i8;
        }
    }

    (int8_data, scales)
}

/// One output row's `f32` value for the INT8 matvec/argmax kernels below:
/// exact `i32` dot product of `x_int8 · w_row` (integer accumulation order
/// never affects the result -- same invariant `avx.rs`'s own doc comments
/// rely on), scaled once at the end. Shared by every INT8 kernel in this
/// module, matching how `avx.rs`'s `int8_row_dot_f32`/`int8_row_dot_argmax`
/// are both thin wrappers around the same per-row dot product.
fn int8_row_dot(w_row: &[i8], x_int8: &[i8], x_scale: f32, w_scale: f32) -> f32 {
    let mut sum: i32 = 0;
    for k in 0..w_row.len() {
        sum += x_int8[k] as i32 * w_row[k] as i32;
    }
    sum as f32 * x_scale * w_scale
}

/// INT8 matvec: `y = W_int8 @ x_int8 * (x_scale * w_scales[row])`, optionally
/// plus `bias`. Portable equivalent of `avx::matvec_int8`.
///
/// # Safety
/// `x_int8` must point to at least `in_dim` valid bytes; `w_int8` to at least
/// `out_dim * in_dim`; `bias`, if present, to at least `out_dim` elements.
#[allow(clippy::too_many_arguments)]
pub unsafe fn matvec_int8(
    y: &mut [f32],
    x_int8: *const i8,
    x_scale: f32,
    w_int8: *const i8,
    w_scales: &[f32],
    bias: Option<&[f32]>,
    in_dim: usize,
    out_dim: usize,
) {
    let x = unsafe { std::slice::from_raw_parts(x_int8, in_dim) };
    for o in 0..out_dim {
        let w_row = unsafe { std::slice::from_raw_parts(w_int8.add(o * in_dim), in_dim) };
        let mut val = int8_row_dot(w_row, x, x_scale, w_scales[o]);
        if let Some(b) = bias {
            val += b[o];
        }
        y[o] = val;
    }
}

/// Find the argmax over rows `[start, end)` of the INT8-quantized `x @ W.T`,
/// returning `(row_index, score)`. Portable equivalent of
/// `avx::argmax_int8_range`.
///
/// # Safety
/// `x_int8` must point to at least `in_dim` valid bytes; `w_int8` to at
/// least `end * in_dim`.
pub unsafe fn argmax_int8_range(
    x_int8: *const i8,
    x_scale: f32,
    w_int8: *const i8,
    w_scales: &[f32],
    in_dim: usize,
    start: usize,
    end: usize,
) -> (usize, f32) {
    let x = unsafe { std::slice::from_raw_parts(x_int8, in_dim) };
    let mut best = start;
    let mut best_val = -1e30f32;
    for o in start..end {
        let w_row = unsafe { std::slice::from_raw_parts(w_int8.add(o * in_dim), in_dim) };
        let val = int8_row_dot(w_row, x, x_scale, w_scales[o]);
        if val > best_val {
            best_val = val;
            best = o;
        }
    }
    (best, best_val)
}

/// Batched INT8 matvec: for each output row, apply it to all `b` sessions
/// (each with its own quantized input + scale + output + optional residual
/// bias). Row-`o`, session-`bi` output equals [`matvec_int8`]'s row-`o`
/// output for that session. Portable equivalent of `avx::matvec_int8_batched`.
///
/// # Safety
/// All pointers must be valid for the stated ranges; `y`, `x_int8`,
/// `x_scale`, and `bias` (if present) must each have length `b`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn matvec_int8_batched(
    b: usize,
    y: &[*mut f32],
    x_int8: &[*const i8],
    x_scale: &[f32],
    w_int8: *const i8,
    w_scales: &[f32],
    bias: Option<&[*const f32]>,
    in_dim: usize,
    out_dim: usize,
) {
    for (o, &ws) in w_scales.iter().enumerate().take(out_dim) {
        let w_row = unsafe { std::slice::from_raw_parts(w_int8.add(o * in_dim), in_dim) };
        for bi in 0..b {
            let x = unsafe { std::slice::from_raw_parts(x_int8[bi], in_dim) };
            let mut val = int8_row_dot(w_row, x, x_scale[bi], ws);
            if let Some(bs) = bias {
                val += unsafe { *bs[bi].add(o) };
            }
            unsafe {
                *y[bi].add(o) = val;
            }
        }
    }
}

/// Batched fused gate_up + SwiGLU: for each intermediate row `j`, compute
/// `SiLU(gate) * up` for all `b` sessions, where `gate`/`up` come from INT8
/// weight rows `2j`/`2j+1`. Portable equivalent of `avx::swiglu_int8_batched`.
///
/// # Safety
/// All pointers must be valid for the stated ranges; `ffn`, `x_int8`, and
/// `x_scale` must each have length `b`; `w_scales` at least `2 * n_rows`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn swiglu_int8_batched(
    b: usize,
    ffn: &[*mut f32],
    x_int8: &[*const i8],
    x_scale: &[f32],
    w_int8: *const i8,
    w_scales: &[f32],
    in_dim: usize,
    n_rows: usize,
) {
    for j in 0..n_rows {
        let wg = unsafe { std::slice::from_raw_parts(w_int8.add(2 * j * in_dim), in_dim) };
        let wu = unsafe { std::slice::from_raw_parts(w_int8.add((2 * j + 1) * in_dim), in_dim) };
        let sg = w_scales[2 * j];
        let su = w_scales[2 * j + 1];
        for bi in 0..b {
            let x = unsafe { std::slice::from_raw_parts(x_int8[bi], in_dim) };
            let g = int8_row_dot(wg, x, x_scale[bi], sg);
            let u = int8_row_dot(wu, x, x_scale[bi], su);
            unsafe {
                *ffn[bi].add(j) = g / (1.0 + (-g).exp()) * u;
            }
        }
    }
}

/// Batched INT8 argmax (lm_head): stream each weight row of `[start, end)`
/// once and update every session's running `(best, best_val)`, with
/// index-stable tie-breaking (strict `>`, lowest row index wins ties).
/// `best`/`best_val` are per-session running state (caller initializes to
/// `0`/`-1e30` before the first call across a disjoint row range). Portable
/// equivalent of `avx::argmax_int8_batched`.
///
/// # Safety
/// All pointers must be valid for the stated ranges; `best`, `best_val`,
/// `x_int8`, and `x_scale` must each have length `b`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn argmax_int8_batched(
    b: usize,
    best: &mut [usize],
    best_val: &mut [f32],
    x_int8: &[*const i8],
    x_scale: &[f32],
    w_int8: *const i8,
    w_scales: &[f32],
    in_dim: usize,
    start: usize,
    end: usize,
) {
    for (o, &ws) in w_scales.iter().enumerate().take(end).skip(start) {
        let w_row = unsafe { std::slice::from_raw_parts(w_int8.add(o * in_dim), in_dim) };
        for bi in 0..b {
            let x = unsafe { std::slice::from_raw_parts(x_int8[bi], in_dim) };
            let val = int8_row_dot(w_row, x, x_scale[bi], ws);
            if val > best_val[bi] {
                best_val[bi] = val;
                best[bi] = o;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_to_f32_buf_matches_scalar_conversion_per_element() {
        let src: [u16; 4] = [0x3F80, 0xBF80, 0x4000, 0x0000]; // 1.0, -1.0, 2.0, 0.0
        let mut dst = [0.0f32; 4];
        bf16_to_f32_buf(&mut dst, &src);
        for (i, &s) in src.iter().enumerate() {
            assert_eq!(dst[i], bf16_to_f32(s));
        }
        assert_eq!(dst, [1.0, -1.0, 2.0, 0.0]);
    }

    #[test]
    fn rms_norm_row_of_a_constant_vector_is_the_weight_itself() {
        // x = [2, 2, 2, 2] -> mean(x^2) = 4, rms = 2 (eps negligible) ->
        // out = x / 2 * weight = weight.
        let x = [2.0f32; 4];
        let weight = [1.0, 2.0, 3.0, 4.0];
        let mut out = [0.0f32; 4];
        rms_norm_row(&mut out, &x, &weight, 4, 1e-6);
        for i in 0..4 {
            assert!((out[i] - weight[i]).abs() < 1e-4, "out[{i}]={} weight[{i}]={}", out[i], weight[i]);
        }
    }

    #[test]
    fn rms_norm_row_matches_hand_computed_values() {
        // x = [3, 4], mean(x^2) = (9+16)/2 = 12.5, rms = sqrt(12.5) ~ 3.5355
        let x = [3.0f32, 4.0];
        let weight = [1.0f32, 1.0];
        let mut out = [0.0f32; 2];
        rms_norm_row(&mut out, &x, &weight, 2, 0.0);
        let rms = 12.5f32.sqrt();
        assert!((out[0] - 3.0 / rms).abs() < 1e-5);
        assert!((out[1] - 4.0 / rms).abs() < 1e-5);
    }

    #[test]
    fn layer_norm_row_of_a_constant_vector_is_just_bias() {
        // x constant -> mean == x, variance == 0 -> normalized == 0 -> out == bias.
        let x = [5.0f32; 4];
        let weight = [1.0, 2.0, 3.0, 4.0];
        let bias = [10.0, 20.0, 30.0, 40.0];
        let mut out = [0.0f32; 4];
        layer_norm_row(&mut out, &x, &weight, &bias, 4, 1e-6);
        for i in 0..4 {
            assert!((out[i] - bias[i]).abs() < 1e-2, "out[{i}]={} bias[{i}]={}", out[i], bias[i]);
        }
    }

    #[test]
    fn layer_norm_row_matches_hand_computed_values() {
        // x = [1, 2, 3, 4], mean = 2.5, var = mean((x-mean)^2) = (2.25+0.25+0.25+2.25)/4 = 1.25
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let weight = [1.0f32; 4];
        let bias = [0.0f32; 4];
        let mut out = [0.0f32; 4];
        layer_norm_row(&mut out, &x, &weight, &bias, 4, 0.0);
        let inv_std = 1.0 / 1.25f32.sqrt();
        let expected = [
            (1.0 - 2.5) * inv_std,
            (2.0 - 2.5) * inv_std,
            (3.0 - 2.5) * inv_std,
            (4.0 - 2.5) * inv_std,
        ];
        for i in 0..4 {
            assert!((out[i] - expected[i]).abs() < 1e-5, "out[{i}]={} expected[{i}]={}", out[i], expected[i]);
        }
    }

    #[test]
    fn exp_inplace_matches_std_exp_exactly() {
        let mut x = [0.0f32, 1.0, -1.0, 2.5, -3.0];
        let expected: Vec<f32> = x.iter().map(|v| v.exp()).collect();
        exp_inplace(&mut x);
        assert_eq!(&x[..], &expected[..]);
    }

    #[test]
    fn exp_inplace_of_zero_is_one() {
        let mut x = [0.0f32];
        exp_inplace(&mut x);
        assert_eq!(x[0], 1.0);
    }

    #[test]
    fn gelu_inplace_of_zero_is_zero() {
        let mut x = [0.0f32];
        gelu_inplace(&mut x, 1);
        assert_eq!(x[0], 0.0);
    }

    #[test]
    fn gelu_inplace_matches_hand_computed_tanh_approximation() {
        let mut x = [1.0f32, -1.0, 2.0];
        gelu_inplace(&mut x, 3);
        // GELU(1) ~ 0.8412, GELU(-1) ~ -0.1588, GELU(2) ~ 1.9546 (tanh approx, standard reference values)
        assert!((x[0] - 0.8412).abs() < 1e-3, "GELU(1)={}", x[0]);
        assert!((x[1] - (-0.1588)).abs() < 1e-3, "GELU(-1)={}", x[1]);
        assert!((x[2] - 1.9546).abs() < 1e-3, "GELU(2)={}", x[2]);
    }

    #[test]
    fn gelu_inplace_is_monotonically_increasing_for_positive_inputs() {
        let mut x = [0.5f32, 1.0, 1.5, 2.0, 2.5];
        gelu_inplace(&mut x, 5);
        for i in 1..x.len() {
            assert!(x[i] > x[i - 1], "GELU should be increasing here: {:?}", x);
        }
    }

    #[test]
    fn quantize_bf16_to_int8_round_trips_within_one_quantization_step() {
        // 4 BF16 values in one row: [1.0, -2.0, 0.5, 4.0] -> max_abs = 4.0
        let bf16_vals: [u16; 4] = [0x3F80, 0xC000, 0x3F00, 0x4080]; // 1.0, -2.0, 0.5, 4.0
        let (int8_data, scales) =
            unsafe { quantize_bf16_to_int8(bf16_vals.as_ptr(), 1, 4) };
        assert_eq!(scales.len(), 1);
        assert!((scales[0] - 4.0 / 127.0).abs() < 1e-6);
        // Dequantize and compare within one quantization step (scale).
        let original = [1.0f32, -2.0, 0.5, 4.0];
        for k in 0..4 {
            let dequant = int8_data[k] as f32 * scales[0];
            assert!(
                (dequant - original[k]).abs() <= scales[0] + 1e-6,
                "k={k} dequant={dequant} original={} scale={}",
                original[k],
                scales[0]
            );
        }
        // The max-magnitude element should quantize to exactly +-127.
        assert_eq!(int8_data[3], 127); // 4.0 is the max
    }

    #[test]
    fn quantize_bf16_to_int8_all_zero_row_produces_zero_scale_safe_default() {
        let bf16_vals: [u16; 2] = [0x0000, 0x0000]; // 0.0, 0.0
        let (int8_data, scales) = unsafe { quantize_bf16_to_int8(bf16_vals.as_ptr(), 1, 2) };
        assert_eq!(scales[0], 1.0); // max_abs == 0.0 -> scale defaults to 1.0, not NaN/inf
        assert_eq!(int8_data, vec![0, 0]);
    }

    #[test]
    fn matvec_int8_matches_hand_computed_dot_product() {
        // x = [1, 2, 3], w row0 = [1, 1, 1] -> dot = 1+2+3 = 6
        //                w row1 = [2, 0, -1] -> dot = 2+0-3 = -1
        let x_int8: [i8; 3] = [1, 2, 3];
        let w_int8: [i8; 6] = [1, 1, 1, 2, 0, -1];
        let w_scales = [1.0f32, 1.0];
        let mut y = [0.0f32; 2];
        unsafe {
            matvec_int8(&mut y, x_int8.as_ptr(), 1.0, w_int8.as_ptr(), &w_scales, None, 3, 2);
        }
        assert_eq!(y[0], 6.0);
        assert_eq!(y[1], -1.0);
    }

    #[test]
    fn matvec_int8_applies_scales_and_bias() {
        let x_int8: [i8; 2] = [10, 10];
        let w_int8: [i8; 2] = [1, 1];
        let w_scales = [0.5f32];
        let bias = [3.0f32];
        let mut y = [0.0f32; 1];
        unsafe {
            matvec_int8(&mut y, x_int8.as_ptr(), 0.1, w_int8.as_ptr(), &w_scales, Some(&bias), 2, 1);
        }
        // dot = 10*1 + 10*1 = 20; 20 * 0.1 * 0.5 = 1.0; + bias 3.0 = 4.0
        assert_eq!(y[0], 4.0);
    }

    #[test]
    fn argmax_int8_range_finds_the_highest_scoring_row() {
        let x_int8: [i8; 2] = [1, 1];
        // row0 dot = 1+1=2, row1 dot = 3+3=6 (highest), row2 dot = 0+0=0
        let w_int8: [i8; 6] = [1, 1, 3, 3, 0, 0];
        let w_scales = [1.0f32, 1.0, 1.0];
        let (best, best_val) =
            unsafe { argmax_int8_range(x_int8.as_ptr(), 1.0, w_int8.as_ptr(), &w_scales, 2, 0, 3) };
        assert_eq!(best, 1);
        assert_eq!(best_val, 6.0);
    }

    #[test]
    fn argmax_int8_range_ties_prefer_the_lowest_index() {
        let x_int8: [i8; 1] = [1];
        let w_int8: [i8; 2] = [5, 5]; // both rows score identically
        let w_scales = [1.0f32, 1.0];
        let (best, _) =
            unsafe { argmax_int8_range(x_int8.as_ptr(), 1.0, w_int8.as_ptr(), &w_scales, 1, 0, 2) };
        assert_eq!(best, 0);
    }

    #[test]
    fn matvec_int8_batched_matches_single_session_matvec_per_session() {
        let in_dim = 3;
        let out_dim = 2;
        let b = 2;
        let w_int8: [i8; 6] = [1, 1, 1, 2, 0, -1];
        let w_scales = [1.0f32, 1.0];

        let x0: [i8; 3] = [1, 2, 3];
        let x1: [i8; 3] = [4, 5, 6];
        let x_scale = [1.0f32, 1.0];
        let x_int8_ptrs = [x0.as_ptr(), x1.as_ptr()];

        let mut y0 = [0.0f32; 2];
        let mut y1 = [0.0f32; 2];
        let y_ptrs = [y0.as_mut_ptr(), y1.as_mut_ptr()];

        unsafe {
            matvec_int8_batched(
                b, &y_ptrs, &x_int8_ptrs, &x_scale, w_int8.as_ptr(), &w_scales, None, in_dim, out_dim,
            );
        }

        // Cross-check against the single-session matvec_int8 for each session.
        let mut expected0 = [0.0f32; 2];
        let mut expected1 = [0.0f32; 2];
        unsafe {
            matvec_int8(&mut expected0, x0.as_ptr(), 1.0, w_int8.as_ptr(), &w_scales, None, in_dim, out_dim);
            matvec_int8(&mut expected1, x1.as_ptr(), 1.0, w_int8.as_ptr(), &w_scales, None, in_dim, out_dim);
        }
        assert_eq!(y0, expected0);
        assert_eq!(y1, expected1);
    }

    #[test]
    fn swiglu_int8_batched_matches_hand_computed_silu_gate() {
        // in_dim=1, n_rows=1 (one gate row, one up row), b=1
        // gate weight row = [2], up weight row = [3], x = [1]
        // g = 1*2*1.0*1.0 = 2.0, u = 1*3*1.0*1.0 = 3.0
        // SiLU(2.0) = 2.0 / (1 + exp(-2.0)) ~ 1.7616
        let x0: [i8; 1] = [1];
        let x_int8_ptrs = [x0.as_ptr()];
        let x_scale = [1.0f32];
        let w_int8: [i8; 2] = [2, 3]; // row0 = gate, row1 = up
        let w_scales = [1.0f32, 1.0];
        let mut ffn0 = [0.0f32; 1];
        let ffn_ptrs = [ffn0.as_mut_ptr()];

        unsafe {
            swiglu_int8_batched(1, &ffn_ptrs, &x_int8_ptrs, &x_scale, w_int8.as_ptr(), &w_scales, 1, 1);
        }

        let g = 2.0f32;
        let u = 3.0f32;
        let expected = g / (1.0 + (-g).exp()) * u;
        assert!((ffn0[0] - expected).abs() < 1e-5, "got {} expected {}", ffn0[0], expected);
    }

    #[test]
    fn argmax_int8_batched_matches_single_session_argmax_per_session() {
        let in_dim = 2;
        let b = 2;
        let w_int8: [i8; 6] = [1, 1, 3, 3, 0, 0]; // 3 rows
        let w_scales = [1.0f32, 1.0, 1.0];

        let x0: [i8; 2] = [1, 1];
        let x1: [i8; 2] = [1, 0]; // row scores differ for this session: row0=1,row1=3,row2=0
        let x_scale = [1.0f32, 1.0];
        let x_int8_ptrs = [x0.as_ptr(), x1.as_ptr()];

        let mut best = [0usize; 2];
        let mut best_val = [-1e30f32; 2];

        unsafe {
            argmax_int8_batched(
                b, &mut best, &mut best_val, &x_int8_ptrs, &x_scale, w_int8.as_ptr(), &w_scales, in_dim, 0, 3,
            );
        }

        assert_eq!(best[0], 1); // row1 wins for both sessions here (3,3 beats 1,1 and 0,0)
        assert_eq!(best[1], 1);
    }
}
