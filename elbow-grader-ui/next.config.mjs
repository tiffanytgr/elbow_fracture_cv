/** @type {import('next').NextConfig} */
const nextConfig = {
  // The FastAPI backend URL — override with BACKEND_URL env var in production.
  env: {
    BACKEND_URL: process.env.BACKEND_URL ?? "http://localhost:8000",
  },
};

export default nextConfig;
