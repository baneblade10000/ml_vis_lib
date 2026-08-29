# Neural Network Playground - Rust + WebAssembly

🧠 Interactive neural network training playground in your browser using Rust and WebAssembly with Next.js

## 🌐 Live Demo

Visit the live demo: [https://nktr-cp.github.io/multilayer-perceptron/](https://nktr-cp.github.io/multilayer-perceptron/)

## ✨ Features

### 🏗️ Architecture & Performance
- **Clean Architecture Core**: Domain-driven design with clear separation of concerns (domain/usecase/adapters)
- **WebAssembly Performance**: High-performance neural network computation using Rust compiled to WebAssembly
- **Modern Frontend**: React/Next.js with TypeScript for type-safe, responsive user interface
- **Real-time Visualization**: Live network architecture visualization with color-coded weights

### 🎯 Machine Learning Capabilities
- **Multiple Task Types**: Binary classification, multi-class classification, and regression with automatic configuration
- **Advanced Training Features**: Early stopping, multiple optimizers (Adam, SGD, RMSProp), regularization (L1/L2)
- **Dataset Generation**: Built-in synthetic dataset generators (XOR, spiral, circular, diagonal patterns)
- **Interactive Training**: Real-time loss/accuracy charts, training progress tracking, and network state inspection

### 🎨 User Experience
- **Professional UI**: Modern design with Tailwind CSS and Framer Motion animations
- **Educational Focus**: Color legends, training logs, and detailed metrics for learning
- **Cross-platform**: Runs entirely in the browser with no server requirements

## 🚀 Quick Start

### Prerequisites

- [Rust](https://rustup.rs/) (1.80.0 or later)
- [Node.js](https://nodejs.org/) (18.0.0 or later)
- [wasm-bindgen-cli](https://rustwasm.github.io/wasm-bindgen/reference/cli.html)

### Installation

```bash
# Clone the repository
git clone https://github.com/nktr-cp/multilayer-perceptron.git
cd multilayer-perceptron

# Install dependencies
make install

# Build WASM module and start development server
make dev
```

### Production Build

```bash
# Build everything for production deployment
make build

# Build and deploy to GitHub Pages
make deploy
```

## 🏗️ Architecture Overview

### Frontend (Next.js + TypeScript)
- **React Components**: Interactive UI with real-time updates
- **Custom Hooks**: `useWASM()` for module loading, `useMLTraining()` for training state
- **Visualization**: Chart.js for metrics, Canvas API for network diagrams
- **Styling**: Tailwind CSS with custom animations

### Backend (Rust + WASM)
- **Core ML Library**: Pure Rust neural network implementation
- **WASM Bindings**: JavaScript-friendly API with `wasm-bindgen`
- **Performance**: Optimized tensor operations and training loops
- **Cross-platform**: Compiles to both native and WebAssembly targets

## 🎯 Usage Examples

### Web Interface
1. **Generate Dataset**: Choose from XOR, spiral, circular, or custom patterns
2. **Configure Network**: Set layer sizes, activation functions, and hyperparameters  
3. **Train Model**: Monitor progress with real-time loss/accuracy charts
4. **Visualize Results**: See decision boundaries and network weights

### Rust API
```rust
use multilayer_perceptron::prelude::*;

// Create a binary classifier
let mut model = Sequential::new()
    .dense(2, 64, Activation::ReLU, WeightInit::XavierUniform)
    .dense(64, 32, Activation::ReLU, WeightInit::XavierUniform)  
    .dense(32, 1, Activation::Sigmoid, WeightInit::XavierUniform);

// Configure training
let config = TrainingConfig {
    epochs: 100,
    batch_size: 32,
    learning_rate: 0.01,
    enable_early_stopping: true,
    early_stopping_patience: 10,
    ..Default::default()
};

// Train the model
let mut trainer = Trainer::new(&mut model, BinaryCrossEntropy::new(), Adam::new(0.01))
    .with_config(config);
    
let history = trainer.fit(&train_x, &train_y, Some(&val_x), Some(&val_y))?;
let mut train_usecase = TrainMLPUsecase::new(data_repo.clone());
let response = train_usecase.execute(request)?;
```

`TrainMLPUsecase` will adjust the model's output activation, choose a loss/metric bundle, and ensure the preprocessing pipeline is applied consistently across training and validation datasets.

## 📦 Building and Deployment

### Local Development

```bash
# Development server
make dev                    # Start dev server at http://localhost:8080

# Production build
make build-pages           # Build optimized version for GitHub Pages
make deploy-local          # Test production build locally at http://localhost:8000
```

### GitHub Pages Deployment

This project is configured for automatic deployment to GitHub Pages:

1. **Automatic Deployment**: 
   - Push to `main` branch triggers automatic deployment via GitHub Actions
   - The workflow builds the optimized WebAssembly and creates static files
   - GitHub Pages serves the content from the `dist/` directory

2. **Manual Deployment**:
   ```bash
   make prepare-deploy     # Build and prepare files for deployment
   git add .
   git commit -m "Deploy to GitHub Pages"
   git push origin main    # Triggers automatic deployment
   ```

3. **Local Testing**:
   ```bash
make deploy-local       # Test the GitHub Pages build locally
```

### Build Optimization

## 🛠️ Build System & Deployment

### Local Development
```bash
make dev          # Start development server with hot reload
make build        # Build for production  
make test-all     # Run all tests (Rust + WASM + Frontend)
make clean        # Clean all build artifacts
```

### CI/CD Pipeline
The project uses GitHub Actions for automated deployment:
- **WASM Build**: Compiles Rust to WebAssembly using manual build commands
- **Frontend Build**: Next.js static export for GitHub Pages compatibility
- **Deployment**: Automatic deployment to GitHub Pages on main branch

### Build Optimizations
- **WASM Size**: Uses `opt-level = "s"` and manual wasm-bindgen for size optimization
- **Next.js**: Static export with proper asset paths for GitHub Pages
- **Performance**: Lazy loading, code splitting, and optimized bundle sizes

## 🖥️ Native Training Demos

Run the CLI examples to see the training system in action:

```bash
# Interactive training demo with console output
cargo run --example training_demo

# Compare different optimizers
cargo run --example optimizer_comparison  

# Demonstrate regularization techniques
cargo run --example regularization_demo

# Enable GUI plots on native (desktop only)
SHOW_GUI_PLOTS=1 cargo run --example training_demo
```

The native examples demonstrate the full Rust API and can open GUI windows with real-time training visualizations when `SHOW_GUI_PLOTS=1` is set.

## 🏗️ Project Structure

```
├── src/                    # Rust backend
│   ├── core/              # Tensor operations and computation graph
│   ├── domain/            # ML models, losses, metrics (business logic)
│   ├── usecase/           # Training, preprocessing, evaluation services  
│   ├── adapters/          # Data loaders, WASM bindings, presentations
│   └── app/               # High-level API and configuration
├── frontend/              # Next.js frontend application
│   ├── app/              # Next.js app router
│   ├── components/       # React components
│   ├── hooks/           # Custom React hooks
│   └── public/          # Static assets
├── examples/             # Rust CLI examples
├── data/                # Sample datasets
└── .github/workflows/   # CI/CD pipeline
```

### Clean Architecture Flow

The project follows clean architecture principles:

```
CLI/Web UI ──▶ app ──▶ usecase ──▶ domain ──▶ core
                 ▲        ▲         ▲
                 │        │         │
            adapters ─────┴─────────┘
```

- **Core**: Pure mathematical operations (tensors, graphs)
- **Domain**: Business logic (models, training, metrics)  
- **Usecase**: Application services orchestrating domain logic
- **Adapters**: External interfaces (WASM, data loading, UI)
- **App**: Configuration and high-level coordination

Each outer layer depends only on the layer directly beneath it. Adapters implement the `domain::ports` traits, while use cases orchestrate datasets, preprocessing pipelines, optimisers, and task-aware strategy selection.

## 🧪 Testing

```bash
# Run all tests
make test-all

# Rust tests
make test-rust
cargo test

# WebAssembly tests
## 🔧 Development Commands

| Command | Description |
|---------|-------------|
| `make help` | Show all available commands |
| `make install` | Install all dependencies (Rust + Node.js) |
| `make dev` | Start development server with hot reload |
| `make build` | Build everything for production |
| `make deploy` | Build for GitHub Pages deployment |
| `make clean` | Clean all build artifacts |
| `make test-all` | Run all tests (Rust + WASM + Frontend) |
| `make lint` | Lint all code (Rust + TypeScript) |
| `make format` | Format all code |
| `make audit` | Check for security vulnerabilities |
| `make serve-prod` | Serve production build locally |

## 📊 Performance & Technical Details

### Bundle Optimization
- **WASM Module**: ~150KB (optimized with wasm-bindgen + manual compilation)  
- **Frontend Bundle**: Code splitting with Next.js for optimal loading
- **Total Load Time**: <3 seconds on typical connections
- **Runtime Performance**: 60 FPS real-time training visualization

### Browser Compatibility  
- **Modern Browsers**: Chrome 67+, Firefox 61+, Safari 11+, Edge 79+
- **WASM Support**: All modern browsers with WebAssembly support
- **Responsive Design**: Works on desktop, tablet, and mobile devices

### Technical Stack
- **Backend**: Rust 1.80+ with custom tensor library and autodiff
- **WASM**: wasm-bindgen for JavaScript interop
- **Frontend**: Next.js 14 + TypeScript + Tailwind CSS
- **Charts**: Chart.js for real-time training metrics
- **Animation**: Framer Motion for smooth UI transitions

## 🤝 Contributing

We welcome contributions! Please read our [contributing guidelines](CONTRIBUTING.md) before submitting PRs.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with proper tests
4. Run the full test suite (`make test-all`)
5. Format and lint your code (`make format && make lint`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Development Setup
```bash
git clone https://github.com/nktr-cp/multilayer-perceptron.git
cd multilayer-perceptron
make install    # Install all dependencies
make dev        # Start development
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [wasm-bindgen](https://github.com/rustwasm/wasm-bindgen) for excellent Rust-WASM interop
- [Next.js](https://nextjs.org/) for the powerful React framework
- [Chart.js](https://www.chartjs.org/) for beautiful data visualization
- The Rust community for outstanding documentation and tooling

---

Built with ❤️ using Rust 🦀 and WebAssembly 🕸️
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT OR Apache-2.0 License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [wasm-pack](https://rustwasm.github.io/wasm-pack/) for WebAssembly tooling
- [Chart.js](https://www.chartjs.org/) for visualization
- [ndarray](https://github.com/rust-ndarray/ndarray) for tensor operations
- [Rust WebAssembly Book](https://rustwasm.github.io/docs/book/) for guidance

## 🔗 Links

- [Live Demo](https://nktr-cp.github.io/multilayer-perceptron/)
- [GitHub Repository](https://github.com/nktr-cp/multilayer-perceptron)
- [Documentation](https://docs.rs/multilayer-perceptron)
- [Issues](https://github.com/nktr-cp/multilayer-perceptron/issues)

---

Made with ❤️ using Rust 🦀 and WebAssembly 🕷️
