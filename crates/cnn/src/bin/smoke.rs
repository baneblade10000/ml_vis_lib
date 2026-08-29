//! Native smoke for PlaygroundEngine (CPU / optional GPU).

use cnn::config::CnnConfig;
use cnn::engine::PlaygroundEngine;
use cnn::gpu::GpuContext;
use cnn::IMAGE_SIZE;

fn main() {
    let mut cfg = CnnConfig::default();
    cfg.noise = 0.05;
    let gpu = GpuContext::try_init_blocking();
    let backend = if gpu.is_some() { "gpu" } else { "cpu" };
    let mut eng = PlaygroundEngine::new_with_gpu(cfg, gpu);
    println!("cnn playground smoke backend={backend} ({})", eng.backend_name);

    let n = 160usize;
    let mut flat = vec![0.0f32; n * IMAGE_SIZE * IMAGE_SIZE];
    let mut labels = vec![0i32; n];
    for i in 0..n {
        labels[i] = if i % 2 == 0 { 0 } else { 1 };
        let base = i * IMAGE_SIZE * IMAGE_SIZE;
        for p in 0..(IMAGE_SIZE * IMAGE_SIZE) {
            // Separable pattern so the net can actually learn something.
            let r = p / IMAGE_SIZE;
            let c = p % IMAGE_SIZE;
            flat[base + p] = if labels[i] == 0 {
                if (r as i32 - 8).pow(2) + (c as i32 - 8).pow(2) < 25 {
                    1.0
                } else {
                    0.0
                }
            } else if r > 4 && r < 12 && c > 4 && c < 12 {
                1.0
            } else {
                0.0
            };
        }
    }
    eng.set_data_2d(&flat, &labels, n, n / 2);

    for _ in 0..20 {
        eng.train_epoch(true);
        if eng.epoch % 5 == 0 {
            eng.refresh_metrics();
            let s = eng.stats();
            println!(
                "epoch {} loss_train={:.4} loss_test={:.4} acc={:.1}%/{:.1}%",
                s.epoch,
                s.loss_train,
                s.loss_test,
                s.acc_train * 100.0,
                s.acc_test * 100.0
            );
        }
    }
}
