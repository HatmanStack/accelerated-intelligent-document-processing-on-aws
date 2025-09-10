/// <reference types="vitest" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    host: true,
    open: true
  },
  build: {
    outDir: 'build',
    sourcemap: true
  },
  define: {
    global: 'globalThis',
  },
  resolve: {
    alias: {
      './runtimeConfig': './runtimeConfig.browser',
      buffer: 'buffer',
    }
  },
  optimizeDeps: {
    include: ['aws-amplify', '@aws-amplify/ui-react', 'buffer']
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './src/setupTests.js',
    css: true,
    server: {
      deps: {
        external: ['maplibre-gl']
      }
    }
  }
})