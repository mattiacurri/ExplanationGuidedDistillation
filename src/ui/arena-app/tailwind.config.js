/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
      },
      colors: {
        surface: {
          DEFAULT: '#ffffff',
          muted: '#f8fafc',
          strong: '#f1f5f9',
        },
        accent: {
          DEFAULT: '#6366f1',
          hover: '#4f46e5',
          subtle: '#eef2ff',
        },
        success: {
          DEFAULT: '#10b981',
          subtle: '#ecfdf5',
        },
      },
    },
  },
  plugins: [],
};
