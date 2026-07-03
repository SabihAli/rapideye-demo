import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from 'react'
import { getAllModelSwitches, updateModelSwitches } from '@/lib/api'
import type { CameraModelSwitches } from '@/types/api'

type SwitchPatch = Partial<Pick<CameraModelSwitches, 'fire_enabled' | 'weapon_enabled' | 'face_enabled'>>

interface ModelSwitchesContextValue {
  loading: boolean
  error: string | null
  getSwitches: (cameraId: number) => CameraModelSwitches | null
  isUpdating: (cameraId: number) => boolean
  setSwitch: (cameraId: number, patch: SwitchPatch) => void
  refresh: () => Promise<void>
}

const ModelSwitchesContext = createContext<ModelSwitchesContextValue | null>(null)
const SAVE_DEBOUNCE_MS = 120

export function ModelSwitchesProvider({ children }: { children: ReactNode }) {
  const [byCamera, setByCamera] = useState<Map<number, CameraModelSwitches>>(new Map())
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [updating, setUpdating] = useState<Set<number>>(new Set())
  const byCameraRef = useRef(byCamera)
  const saveTimersRef = useRef<Map<number, ReturnType<typeof setTimeout>>>(new Map())
  const persistVersionRef = useRef<Map<number, number>>(new Map())

  useEffect(() => {
    byCameraRef.current = byCamera
  }, [byCamera])

  const refresh = useCallback(async () => {
    setLoading(true)
    try {
      const all = await getAllModelSwitches()
      const next = new Map(all.map((row) => [row.camera_id, row]))
      byCameraRef.current = next
      setByCamera(next)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load model switches')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void refresh()
    return () => {
      for (const timer of saveTimersRef.current.values()) {
        clearTimeout(timer)
      }
      saveTimersRef.current.clear()
    }
  }, [refresh])

  const persistCamera = useCallback(async (cameraId: number, version: number) => {
    const latest = byCameraRef.current.get(cameraId)
    if (!latest) return

    setUpdating((prev) => new Set(prev).add(cameraId))
    try {
      const saved = await updateModelSwitches(latest)
      if (persistVersionRef.current.get(cameraId) === version) {
        byCameraRef.current = new Map(byCameraRef.current).set(cameraId, saved)
        setByCamera(new Map(byCameraRef.current))
        setError(null)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update model switches')
    } finally {
      setUpdating((prev) => {
        const copy = new Set(prev)
        copy.delete(cameraId)
        return copy
      })
    }
  }, [])

  const getSwitches = useCallback(
    (cameraId: number) => byCamera.get(cameraId) ?? null,
    [byCamera]
  )

  const isUpdating = useCallback((cameraId: number) => updating.has(cameraId), [updating])

  const setSwitch = useCallback(
    (cameraId: number, patch: SwitchPatch) => {
      const current = byCameraRef.current.get(cameraId)
      if (!current) return

      const next: CameraModelSwitches = { ...current, ...patch, camera_id: cameraId }
      byCameraRef.current = new Map(byCameraRef.current).set(cameraId, next)
      setByCamera(new Map(byCameraRef.current))

      const version = (persistVersionRef.current.get(cameraId) ?? 0) + 1
      persistVersionRef.current.set(cameraId, version)

      const existingTimer = saveTimersRef.current.get(cameraId)
      if (existingTimer) clearTimeout(existingTimer)

      saveTimersRef.current.set(
        cameraId,
        setTimeout(() => {
          saveTimersRef.current.delete(cameraId)
          void persistCamera(cameraId, version)
        }, SAVE_DEBOUNCE_MS)
      )
    },
    [persistCamera]
  )

  return (
    <ModelSwitchesContext.Provider
      value={{ loading, error, getSwitches, isUpdating, setSwitch, refresh }}
    >
      {children}
    </ModelSwitchesContext.Provider>
  )
}

export function useModelSwitches(cameraId: number) {
  const ctx = useContext(ModelSwitchesContext)
  if (!ctx) throw new Error('useModelSwitches must be used within ModelSwitchesProvider')

  const switches = ctx.getSwitches(cameraId)

  return {
    switches,
    loading: ctx.loading,
    updating: ctx.isUpdating(cameraId),
    error: ctx.error,
    setFireEnabled: (enabled: boolean) => ctx.setSwitch(cameraId, { fire_enabled: enabled }),
    setWeaponEnabled: (enabled: boolean) => ctx.setSwitch(cameraId, { weapon_enabled: enabled }),
    setFaceEnabled: (enabled: boolean) => ctx.setSwitch(cameraId, { face_enabled: enabled }),
    refresh: ctx.refresh,
  }
}
