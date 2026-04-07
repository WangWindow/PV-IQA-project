import { defineConfig } from 'vite'
import path from 'node:path'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

const apiPort = Number(process.env.PV_IQA_API_PORT ?? process.env.PORT ?? 6005)

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
      '/api': `http://127.0.0.1:${apiPort}`,
      '/uploads': `http://127.0.0.1:${apiPort}`,
    },
  },
  preview: {
    host: '0.0.0.0',
    port: 6006,
  },
})
