import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from 'react'
import { addCameraByFile, addCameraByUrl, listCameras, removeCamera as apiRemoveCamera } from '@/lib/api'
import { setCameraDefinitions, toCameraDefinition } from '@/lib/cameras'
import type { CameraOut } from '@/types/api'

export const MAX_CAMERAS = 4
const POLL_INTERVAL_MS = 5000

interface CameraContextValue {
  cameras: CameraOut[]
  loading: boolean
  error: string | null
  refresh: () => Promise<void>
  addCameraByUrl: (name: string, url: string) => Promise<CameraOut>
  addCameraByFile: (name: string, file: File, onProgress?: (pct: number) => void) => Promise<CameraOut>
  removeCamera: (cameraId: number) => Promise<void>
}

const CameraContext = createContext<CameraContextValue | null>(null)

function upsert(cameras: CameraOut[], camera: CameraOut): CameraOut[] {
  const withoutCamera = cameras.filter((c) => c.camera_id !== camera.camera_id)
  return [...withoutCamera, camera].sort((a, b) => a.camera_id - b.camera_id)
}

export function CameraProvider({ children }: { children: ReactNode }) {
  const [cameras, setCameras] = useState<CameraOut[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setCameraDefinitions(cameras.map(toCameraDefinition))
  }, [cameras])

  const refresh = useCallback(async () => {
    try {
      const data = await listCameras()
      setCameras(data)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load cameras')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void refresh()
    const timer = window.setInterval(refresh, POLL_INTERVAL_MS)
    return () => window.clearInterval(timer)
  }, [refresh])

  const handleAddByUrl = useCallback(async (name: string, url: string) => {
    const camera = await addCameraByUrl(name, url)
    setCameras((prev) => upsert(prev, camera))
    return camera
  }, [])

  const handleAddByFile = useCallback(
    async (name: string, file: File, onProgress?: (pct: number) => void) => {
      const camera = await addCameraByFile(name, file, onProgress)
      setCameras((prev) => upsert(prev, camera))
      return camera
    },
    []
  )

  const handleRemove = useCallback(async (cameraId: number) => {
    await apiRemoveCamera(cameraId)
    setCameras((prev) => prev.filter((c) => c.camera_id !== cameraId))
  }, [])

  return (
    <CameraContext.Provider
      value={{
        cameras,
        loading,
        error,
        refresh,
        addCameraByUrl: handleAddByUrl,
        addCameraByFile: handleAddByFile,
        removeCamera: handleRemove,
      }}
    >
      {children}
    </CameraContext.Provider>
  )
}

export function useCameras() {
  const ctx = useContext(CameraContext)
  if (!ctx) throw new Error('useCameras must be used within CameraProvider')
  return ctx
}
