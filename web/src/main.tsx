import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { CameraProvider } from '@/context/CameraContext'
import { DemoProvider } from '@/context/DemoContext'
import { ModelSwitchesProvider } from '@/context/ModelSwitchesContext'
import { ZoneProvider } from '@/context/ZoneContext'
import './index.css'
import App from './App.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <CameraProvider>
      <DemoProvider>
        <ModelSwitchesProvider>
          <ZoneProvider>
            <App />
          </ZoneProvider>
        </ModelSwitchesProvider>
      </DemoProvider>
    </CameraProvider>
  </StrictMode>,
)
