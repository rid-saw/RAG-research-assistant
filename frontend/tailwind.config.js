/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        cream: {
          50: "#fefcf6",
          100: "#faf7f2",
          200: "#f0e9da",
          300: "#e8dfcc",
          400: "#d6c9a8",
          500: "#c9b896",
        },
        brown: {
          400: "#9a8567",
          500: "#6e5640",
          600: "#4a3520",
          700: "#3a2f24",
          800: "#2d1f15",
        },
        oxblood: {
          400: "#a64949",
          500: "#7a2e2e",
          600: "#5c2222",
          700: "#421616",
        },
      },
      fontFamily: {
        serif: ['Lora', 'Georgia', 'serif'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
