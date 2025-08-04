import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite';

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: true, // hoặc host: '0.0.0.0' để lắng nghe tất cả interface
    port: 5173, // có thể đổi nếu cần
    // Nếu bạn muốn chỉ định một IP cụ thể:
    // host: '192.168.1.100',
    strictPort: false, // nếu port bận thì tự tăng
    hmr: {
      host: '10.30.3.40', // ví dụ: '192.168.1.100' nếu cần hot-reload hoạt động ổn định từ thiết bị khác
    },
  },
})
