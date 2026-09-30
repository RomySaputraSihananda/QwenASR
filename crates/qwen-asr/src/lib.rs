//! CPU-only Qwen3-ASR speech recognition in pure Rust.
//!
//! BLAS and SIMD optimizations are selected automatically at compile time based
//! on the target platform — Accelerate + NEON on macOS/aarch64, OpenBLAS + AVX2
//! on Linux/x86_64, etc. For best performance on x86_64, build with:
//!
//! ```bash
//! RUSTFLAGS="-C target-cpu=native" cargo build --release
//! ```
//!
//! **Important:** Always build in release mode (`--release`). Debug builds are
//! 10–50x slower and unusable for real-time inference.
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use qwen_asr::context::QwenCtx;
//! use qwen_asr::transcribe;
//!
//! let mut ctx = QwenCtx::load("qwen3-asr-0.6b").expect("model not found");
//! let text = transcribe::transcribe(&mut ctx, "audio.wav").unwrap();
//! println!("{text}");
//! ```
//!
//! # Forced Alignment
//!
//! With the aligner model variant you can obtain word-level timestamps for a
//! known transcript:
//!
//! ```rust,no_run
//! use qwen_asr::context::QwenCtx;
//! use qwen_asr::align;
//!
//! let mut ctx = QwenCtx::load("qwen3-aligner-0.6b").expect("aligner model not found");
//! let samples: Vec<f32> = vec![]; // 16 kHz mono f32 PCM
//! let results = align::forced_align(&mut ctx, &samples, "Hello world", "English").unwrap();
//! for r in &results {
//!     println!("{}: {:.0} – {:.0} ms", r.text, r.start_ms, r.end_ms);
//! }
//! ```
//!
//! # Module Guide
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`context`] | Engine state — start here with [`context::QwenCtx::load`] |
//! | [`transcribe`] | Offline, segmented, and streaming transcription |
//! | [`audio`] | WAV loading, resampling, mel spectrogram |
//! | [`align`] | Forced alignment (word/character timestamps) |
//! | [`config`] | Model configuration and variant detection |
//! | [`tokenizer`] | GPT-2 byte-level BPE tokenizer |
//!
//! The remaining modules (`encoder`, `decoder`, `kernels`, `safetensors`) are
//! implementation details and not intended for direct use.

pub mod align;
pub mod audio;
#[cfg(any(feature = "ios", feature = "android", feature = "macos-ffi"))]
pub mod c_api;
pub mod config;
pub mod context;
pub mod decoder;
pub mod draft;
pub mod encoder;
pub mod int8_sidecar;
#[cfg(feature = "android")]
pub mod jni_api;
pub mod kernels;
pub mod output;
pub mod safetensors;
pub mod subtitle;
pub mod tokenizer;
pub mod transcribe;

/// Returns a list of compile-time optimization flags enabled for this build.
pub fn optimization_flags() -> Vec<&'static str> {
    let mut flags = Vec::new();

    // The `vdsp` feature only reaches real code on Apple platforms; reporting
    // "vDSP/Accelerate" on a Linux build (where `blas` means OpenBLAS) sent
    // issue #47 chasing the wrong library.
    let apple_vdsp = cfg!(feature = "vdsp") && cfg!(target_vendor = "apple");
    if apple_vdsp {
        flags.push("vDSP/Accelerate");
    }
    if cfg!(feature = "blas") && !apple_vdsp {
        flags.push("BLAS");
    }

    // Architecture-specific SIMD
    if cfg!(target_arch = "aarch64") {
        flags.push("NEON");
        if cfg!(target_feature = "dotprod") {
            flags.push("DotProd");
        }
    } else if cfg!(target_arch = "x86_64") {
        if cfg!(target_feature = "avx2") {
            flags.push("AVX2");
        } else if cfg!(target_feature = "avx") {
            flags.push("AVX");
        } else if cfg!(target_feature = "sse4.1") {
            flags.push("SSE4.1");
        }
        if cfg!(target_feature = "fma") {
            flags.push("FMA");
        }
    }

    if flags.is_empty() {
        flags.push("generic");
    }

    flags
}

/// Whether this process's actual CPU has the instruction-set extensions the
/// x86_64 SIMD kernels in [`kernels::avx`](crate::kernels) require (AVX2 +
/// FMA). Always `true` on non-x86_64 targets.
///
/// This is a genuine *runtime* check ([`std::is_x86_feature_detected`]) and
/// is independent of [`optimization_flags`], which only reports what the
/// *compiler* was told to target (`-C target-feature`/`target-cpu`) — not
/// what the CPU this binary actually executes on supports.
///
/// [`kernels::avx`](crate::kernels) is compiled in for every x86_64 build
/// unconditionally (gated on `target_arch`, not `target_feature`), and its
/// functions are marked `#[target_feature(enable = "avx2", enable = "fma")]`
/// — calling one when the CPU lacks those extensions is undefined behavior
/// and crashes the process with `SIGILL`, with no panic or error message.
/// [`context::QwenModel::load`](crate::context::QwenModel::load) checks
/// this before touching any kernel so that gap surfaces as a clear error
/// instead. Confirmed on an Ivy Bridge-era Xeon (AVX present, AVX2/FMA
/// absent — that generation gap is narrow but real: AVX2 only shipped
/// starting with Haswell in 2013, one microarchitecture generation later).
#[cfg(target_arch = "x86_64")]
pub fn cpu_meets_kernel_requirements() -> bool {
    std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
}

/// Always `true` off x86_64 — see the x86_64 overload's doc comment.
#[cfg(not(target_arch = "x86_64"))]
pub fn cpu_meets_kernel_requirements() -> bool {
    true
}
