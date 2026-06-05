/**
 * server.ts — SynFeasNet Frontend Dev Server
 *
 * Simple Vite dev server wrapper. All API calls go through Vite's proxy
 * to the Python FastAPI backend — no Gemini proxy, no fallback molecules.
 */
import express from "express";
import path from "path";
import dotenv from "dotenv";
import { createServer as createViteServer } from "vite";

dotenv.config();

const app = express();
const PORT = parseInt(process.env.PORT || "3000", 10);

// WARNING: Do NOT use app.use(express.json()) here!
// It consumes the request body stream, which causes http-proxy (used by Vite)
// to hang indefinitely when trying to forward POST requests to the FastAPI backend.

async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`SynFeasNet frontend server running on http://localhost:${PORT}`);
    console.log(`API requests are proxied to the FastAPI backend via Vite config.`);
  });
}

startServer();
