# SecureVision AI — Frontend Prototype

Enterprise security command center UI for an AI-powered video surveillance platform.

**Frontend only** — mock data, no backend, no real RTSP streams or biometric data.

## Tech Stack

- React + Vite + TypeScript
- Tailwind CSS v4
- lucide-react icons
- react-router-dom

## Run

```bash
npm install
npm run dev
```

Open http://localhost:5173

## Current Scope

- **Dashboard** — stats, 2×2 live camera cards with AI overlays, alert feed
- **Layout** — sidebar nav, topbar, dark security theme
- Other nav items show placeholder pages (ready for future builds)

## For Backend Partner

Mock data lives in `src/data/mockData.ts`. Replace with API calls when backend is ready.
