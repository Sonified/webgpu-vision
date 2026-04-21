import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  optimizeDeps: {
    exclude: ['onnxruntime-web', 'three'],
  },
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
    fs: {
      allow: ['.'],
    },
  },
  assetsInclude: ['**/*.wasm', '**/*.onnx'],
  build: {
    rollupOptions: {
      external: [/vendor\/onnxruntime-web\/.*/],
    },
  },
});
