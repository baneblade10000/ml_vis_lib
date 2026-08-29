# Neural Network Playground - Rust + WebAssembly + Next.js Build System

.PHONY: all install check-deps build build-wasm build-frontend dev clean test help deploy

# Default target
all: build

# Install all dependencies
install:
	@echo "🔧 Installing dependencies..."
	@echo "📦 Installing frontend dependencies..."
	cd frontend && npm install
	@echo "🦀 Checking Rust toolchain..."
	cargo check
	@echo "✅ Installation complete!"

# Check required dependencies
check-deps:
	@echo "🔍 Checking dependencies..."
	@command -v cargo >/dev/null 2>&1 || (echo "❌ Rust/Cargo not found. Install from https://rustup.rs/" && exit 1)
	@command -v wasm-bindgen >/dev/null 2>&1 || (echo "❌ wasm-bindgen not found. Install with: cargo install wasm-bindgen-cli" && exit 1)
	@command -v node >/dev/null 2>&1 || (echo "❌ Node.js not found. Install from https://nodejs.org/" && exit 1)
	@echo "✅ All dependencies found!"

# Build WebAssembly module
build-wasm: check-deps
	@echo "🚀 Building WebAssembly module..."
	cargo build --target wasm32-unknown-unknown --release
	wasm-bindgen --out-dir pkg --web --typescript target/wasm32-unknown-unknown/release/multilayer_perceptron.wasm
	@echo "✅ WASM build complete!"

# Build Next.js frontend 
build-frontend: build-wasm
	@echo "⚛️  Building Next.js frontend..."
	cd frontend && npm run build
	@echo "✅ Frontend build complete!"

# Build everything for production
build: build-frontend
	@echo "🎉 Production build complete!"

# Development server with hot reload
dev: build-wasm
	@echo "🔥 Starting development server..."
	cd frontend && npm run dev

# Production deployment build
deploy: build-frontend
	@echo "🚀 Building for deployment..."
	cd frontend && npm run export
	@echo "✅ Deployment build ready in frontend/out/"

# Clean build artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	cargo clean
	rm -rf pkg/
	rm -rf target/
	cd frontend && rm -rf .next/ out/ node_modules/.cache/
	@echo "✅ Clean complete!"

# Run Rust tests
test:
	@echo "🧪 Running Rust tests..."
	cargo test
	@echo "✅ Rust tests complete!"

# Run WebAssembly tests  
test-wasm:
	@echo "🌐 Running WebAssembly tests..."
	wasm-pack test --headless --firefox
	@echo "✅ WASM tests complete!"

# Run frontend tests
test-frontend:
	@echo "⚛️  Running frontend tests..."
	cd frontend && npm run test
	@echo "✅ Frontend tests complete!"

# Run all tests
test-all: test test-wasm test-frontend
	@echo "🎉 All tests complete!"

# Lint and format code
lint:
	@echo "🔍 Linting code..."
	cargo clippy -- -D warnings
	cd frontend && npm run lint
	@echo "✅ Linting complete!"

# Format code
format:
	@echo "🎨 Formatting code..."
	cargo fmt
	cd frontend && npm run format
	@echo "✅ Formatting complete!"

# Check for security vulnerabilities
audit:
	@echo "🔒 Auditing dependencies..."
	cargo audit
	cd frontend && npm audit
	@echo "✅ Audit complete!"

# Development utilities
watch:
	@echo "👀 Starting file watcher for Rust code..."
	cargo watch -x check -x test

serve-prod:
	@echo "🌐 Serving production build locally..."
	cd frontend/out && python3 -m http.server 3000

# Help target
help:
	@echo "Neural Network Playground - Build System"
	@echo ""
	@echo "Available targets:"
	@echo "  install       - Install all dependencies"
	@echo "  check-deps    - Check required tools are installed"
	@echo "  build-wasm    - Build WebAssembly module only"
	@echo "  build-frontend- Build Next.js frontend only"  
	@echo "  build         - Build everything for production"
	@echo "  dev           - Start development server"
	@echo "  deploy        - Build for deployment"
	@echo "  clean         - Clean all build artifacts"
	@echo "  test          - Run Rust tests"
	@echo "  test-wasm     - Run WebAssembly tests"
	@echo "  test-frontend - Run frontend tests"
	@echo "  test-all      - Run all tests"
	@echo "  lint          - Lint all code"
	@echo "  format        - Format all code"
	@echo "  audit         - Check for security vulnerabilities"
	@echo "  watch         - Watch Rust files for changes"
	@echo "  serve-prod    - Serve production build locally"
	@echo "  help          - Show this help message"
	npm audit

# Build optimized release
release: clean
	@echo "Building optimized release..."
	CARGO_PROFILE_RELEASE_LTO=true \
	CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1 \
	CARGO_PROFILE_RELEASE_PANIC=abort \
	RUSTFLAGS="--cfg=web_sys_unstable_apis" wasm-pack build --no-opt --target web --out-dir pkg --release
	RUSTFLAGS="--cfg=web_sys_unstable_apis" wasm-pack build --no-opt --target nodejs --out-dir pkg-node --release
	RUSTFLAGS="--cfg=web_sys_unstable_apis" wasm-pack build --no-opt --target bundler --out-dir pkg-bundler --release
	npm run build:demo

# Build for GitHub Pages deployment
build-pages: clean install
	@echo "Building for GitHub Pages deployment..."
	@echo "Building optimized WebAssembly..."
	RUSTFLAGS="--cfg=web_sys_unstable_apis" wasm-pack build --no-opt --target web --out-dir pkg --release
	@echo "Optimizing WebAssembly file size..."
	@if command -v wasm-opt >/dev/null 2>&1; then \
		wasm-opt -Os --enable-bulk-memory --enable-reference-types pkg/multilayer_perceptron_bg.wasm -o pkg/multilayer_perceptron_bg.wasm; \
		echo "WebAssembly optimized with wasm-opt"; \
	else \
		echo "wasm-opt not found, skipping optimization"; \
	fi
	@echo "Building production website..."
	npm run build:pages
	@echo "Build complete! Files are in dist/ directory"

# Deploy to GitHub Pages (local testing)
deploy-local: build-pages
	@echo "Starting local server for GitHub Pages testing..."
	@echo "Open http://localhost:8000 in your browser"
	npm run serve:dist

# Prepare for GitHub Pages deployment
prepare-deploy: build-pages
	@echo "Deployment preparation complete!"
	@echo "Files ready for GitHub Pages in dist/ directory"
	@echo "To deploy manually, push changes and enable GitHub Pages in repository settings"

# Size analysis
analyze:
	@echo "Analyzing WebAssembly bundle size..."
	@if [ -f pkg/multilayer_perceptron_bg.wasm ]; then \
		echo "WebAssembly bundle size:"; \
		ls -lh pkg/multilayer_perceptron_bg.wasm; \
		echo ""; \
		wasm-nm pkg/multilayer_perceptron_bg.wasm | head -20; \
	else \
		echo "WebAssembly bundle not found. Run 'make build-wasm' first."; \
	fi

# Profile build
profile:
	@echo "Building with profiling enabled..."
	CARGO_PROFILE_RELEASE_DEBUG=true wasm-pack build --target web --out-dir pkg --release

# Generate documentation
docs:
	@echo "Generating documentation..."
	cargo doc --no-deps --open

# Check WebAssembly support
check-wasm:
	@echo "Checking WebAssembly toolchain..."
	@rustup target list --installed | grep -q wasm32-unknown-unknown || (echo "Installing wasm32-unknown-unknown target..." && rustup target add wasm32-unknown-unknown)
	@command -v wasm-pack >/dev/null 2>&1 || (echo "Installing wasm-pack..." && curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh)
	@echo "WebAssembly toolchain ready!"

# Help
help:
	@echo "Available targets:"
	@echo "  install       - Install dependencies"
	@echo "  build         - Build WebAssembly modules for all targets"
	@echo "  build-wasm    - Build WebAssembly module for web"
	@echo "  build-node    - Build WebAssembly module for Node.js"
	@echo "  build-bundler - Build WebAssembly module for bundlers"
	@echo "  build-demo    - Build demo website"
	@echo "  build-pages   - Build for GitHub Pages deployment"
	@echo "  dev           - Start development server"
	@echo "  serve         - Start simple HTTP server"
	@echo "  deploy-local  - Test GitHub Pages build locally"
	@echo "  prepare-deploy- Prepare files for GitHub Pages"
	@echo "  check-wasm    - Check WebAssembly toolchain"
	@echo "  test          - Run WebAssembly tests"
	@echo "  test-chrome   - Run tests in Chrome"
	@echo "  test-rust     - Run Rust tests"
	@echo "  test-all      - Run all tests"
	@echo "  clean         - Clean build artifacts"
	@echo "  lint          - Lint Rust code"
	@echo "  format        - Format Rust code"
	@echo "  audit         - Audit dependencies for security"
	@echo "  release       - Build optimized release"
	@echo "  analyze       - Analyze WebAssembly bundle size"
	@echo "  profile       - Build with profiling"
	@echo "  docs          - Generate documentation"
	@echo "  help          - Show this help"
