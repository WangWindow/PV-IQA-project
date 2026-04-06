import { defineConfig } from 'vite'
import path from 'node:path'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    host: '0.0.0.0',
    port: 6006,
    proxy: {
      '/api': 'http://127.0.0.1:6007',
      '/uploads': 'http://127.0.0.1:6007',
    },
  },
  preview: {
    host: '0.0.0.0',
    port: 6006,
  },
})
