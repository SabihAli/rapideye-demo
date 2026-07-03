import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { DemoProvider } from '@/context/DemoContext'
import { ZoneProvider } from '@/context/ZoneContext'
import './index.css'
import App from './App.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <DemoProvider>
      <ZoneProvider>
        <App />
      </ZoneProvider>
    </DemoProvider>
  </StrictMode>,
)
