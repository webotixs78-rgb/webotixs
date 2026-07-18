import type { Config } from 'tailwindcss'

const config: Config = {
  darkMode: 'class',
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        background: {
          DEFAULT: 'var(--bg-primary)',
          secondary: 'var(--bg-secondary)',
          card: 'var(--bg-card)',
          section: 'var(--bg-section)',
        },
        primary: {
          DEFAULT: '#3B82F6',
          hover: '#60A5FA',
          from: '#2563EB',
          to: '#06B6D4',
        },
        border: {
          DEFAULT: 'var(--color-border)',
        },
        text: {
          white: 'var(--color-white)',
          gray: 'var(--color-gray)',
        },
        success: '#22C55E',
      },
      fontFamily: {
        display: ['var(--font-space-grotesk)', 'sans-serif'],
        body: ['var(--font-inter)', 'sans-serif'],
      },
      borderRadius: {
        '2xl': '1rem',
        '3xl': '1.5rem',
        '4xl': '2rem',
      },
      backgroundImage: {
        'gradient-primary': 'linear-gradient(135deg, #2563EB, #06B6D4)',
        'gradient-radial': 'radial-gradient(ellipse at center, var(--tw-gradient-stops))',
        'gradient-mesh': 'radial-gradient(at 40% 20%, #2563EB22 0px, transparent 50%), radial-gradient(at 80% 0%, #06B6D422 0px, transparent 50%), radial-gradient(at 0% 50%, #7C3AED22 0px, transparent 50%)',
        'glow-blue': 'radial-gradient(ellipse 80% 80% at 50% -20%, rgba(59,130,246,0.2), transparent)',
      },
      boxShadow: {
        'glow-sm': '0 0 15px rgba(59,130,246,0.3)',
        'glow-md': '0 0 30px rgba(59,130,246,0.4)',
        'glow-lg': '0 0 60px rgba(59,130,246,0.5)',
        'card': '0 1px 3px rgba(0,0,0,0.4), 0 1px 2px rgba(0,0,0,0.2)',
        'card-hover': '0 10px 40px rgba(0,0,0,0.5), 0 0 20px rgba(59,130,246,0.15)',
      },
      animation: {
        'float': 'float 6s ease-in-out infinite',
        'pulse-slow': 'pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'gradient-x': 'gradient-x 15s ease infinite',
        'shimmer': 'shimmer 2s linear infinite',
        'spin-slow': 'spin 8s linear infinite',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-20px)' },
        },
        'gradient-x': {
          '0%, 100%': { 'background-size': '200% 200%', 'background-position': 'left center' },
          '50%': { 'background-size': '200% 200%', 'background-position': 'right center' },
        },
        shimmer: {
          '0%': { transform: 'translateX(-100%)' },
          '100%': { transform: 'translateX(100%)' },
        },
      },
    },
  },
  plugins: [],
}

export default config
