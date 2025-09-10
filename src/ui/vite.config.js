/// <reference types="vitest" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  esbuild: {
    loader: 'jsx',
    include: /src\/.*\.[jt]sx?$/,
    exclude: []
  },
  optimizeDeps: {
    include: ['aws-amplify', '@aws-amplify/ui-react', 'buffer'],
    esbuildOptions: {
      loader: {
        '.js': 'jsx',
      },
    },
  },
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