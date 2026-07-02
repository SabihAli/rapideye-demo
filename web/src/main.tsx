import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { ZoneProvider } from '@/context/ZoneContext'
import './index.css'
import App from './App.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ZoneProvider>
      <App />
    </ZoneProvider>
  </StrictMode>,
)
