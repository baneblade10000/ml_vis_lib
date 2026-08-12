//! Minimal wgpu context. On wasm → WebGPU; native → Vulkan/Metal/DX12.
//! Heavy CNN math stays on CPU for tiny playground nets; GPU is initialized
//! and ready for GEMM/conv dispatches (hook via `touch` / future shaders).

use std::sync::Arc;

pub struct GpuContext {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub adapter_name: String,
}

impl GpuContext {
    /// Async init — required on wasm (WebGPU).
    pub async fn try_init() -> Option<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            #[cfg(target_arch = "wasm32")]
            backends: wgpu::Backends::BROWSER_WEBGPU,
            #[cfg(not(target_arch = "wasm32"))]
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await?;

        let info = adapter.get_info();
        let adapter_name = format!("{} ({:?})", info.name, info.backend);

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("cnn_wgpu"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_webgl2_defaults()
                        .using_resolution(adapter.limits()),
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .ok()?;

        Some(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter_name,
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn try_init_blocking() -> Option<Self> {
        pollster::block_on(Self::try_init())
    }

    /// Placeholder for future compute dispatches (keeps device/queue live).
    pub fn touch(&self) {
        let _ = (&self.device, &self.queue);
    }
}
