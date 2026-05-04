import express from 'express';
import { createServer as createViteServer } from 'vite';
import path from 'path';
import { fileURLToPath } from 'url';
import { spawn } from 'child_process';
import { createProxyMiddleware } from 'http-proxy-middleware';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Configuration
const PORT = 3000;
const PYTHON_PORT = 5000;
const PYTHON_SCRIPT = 'ai.py';

async function startServer() {
  const app = express();

  // 1. Start Python AI Server
  console.log('Starting Python AI Server...');
  const pythonProcess = spawn('python', [PYTHON_SCRIPT], {
    env: { ...process.env, PYTHON_PORT: PYTHON_PORT.toString(), PYTHONUNBUFFERED: '1' },
    stdio: 'pipe' // Capture stdout/stderr
  });

  pythonProcess.stdout.on('data', (data) => {
    console.log(`[Python AI]: ${data.toString().trim()}`);
  });

  pythonProcess.stderr.on('data', (data) => {
    console.error(`[Python AI Error]: ${data.toString().trim()}`);
  });

  pythonProcess.on('close', (code) => {
    console.log(`Python AI process exited with code ${code}`);
  });

  // 2. Proxy AI Requests to Python Server
  // We use a proxy to forward /api/ai/* to the Python server
  // MUST be before express.json() to avoid body parsing issues
  app.use('/api/ai', createProxyMiddleware({
    target: `http://127.0.0.1:${PYTHON_PORT}`,
    changeOrigin: true,
    pathRewrite: {
      '^/api/ai': '', // Remove /api/ai prefix when forwarding
    },
    onError: (err, req, res) => {
      console.error('Proxy Error:', err);
      res.status(503).json({ error: 'AI Service Unavailable' });
    }
  }));

  // Middleware to parse JSON bodies (for non-proxied routes)
  app.use(express.json());

  // Health Check
  app.get('/api/health', (req, res) => {
    res.json({ status: 'ok' });
  });

  // Vite middleware for development
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    // Production static file serving
    app.use(express.static(path.resolve(__dirname, 'dist')));
    app.get('*', (req, res) => {
      res.sendFile(path.resolve(__dirname, 'dist', 'index.html'));
    });
  }

  const server = app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });

  // Cleanup on exit
  process.on('SIGTERM', () => {
    console.log('SIGTERM received. Cleaning up...');
    pythonProcess.kill();
    server.close();
    process.exit(0);
  });
    
  process.on('SIGINT', () => {
    console.log('SIGINT received. Cleaning up...');
    pythonProcess.kill();
    server.close();
    process.exit(0);
  });
}

startServer();
