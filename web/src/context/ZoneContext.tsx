import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from 'react'
import { getZone, updateZone as apiUpdateZone } from '@/lib/api'
import { cameraFromNumId, cameraNumId } from '@/lib/cameras'
import { useCameras } from '@/context/CameraContext'
import type { ZoneConfig } from '@/types/api'
import type { Point, Zone } from '@/types/zone'
import { ZONE_TYPE_COLORS } from '@/types/zone'

interface ZoneContextValue {
  zones: Zone[]
  loading: boolean
  error: string | null
  getZonesByCamera: (cameraId: string) => Zone[]
  saveCameraZone: (cameraId: string, points: Point[], alertClasses?: string[]) => Promise<void>
  deleteCameraZone: (cameraId: string) => Promise<void>
  reloadZones: () => Promise<void>
}

const ZoneContext = createContext<ZoneContextValue | null>(null)

function normalizedToPoints(polygon: [number, number][]): Point[] {
  return polygon.map(([x, y]) => ({ x: x * 100, y: y * 100 }))
}

function pointsToNormalized(points: Point[]): [number, number][] {
  return points.map((p) => [p.x / 100, p.y / 100])
}

function configToZone(config: ZoneConfig): Zone | null {
  if (!config.polygon.length) return null

  const camera = cameraFromNumId(config.camera_id)

  return {
    id: `zone-cam-${config.camera_id}`,
    cameraId: camera.id,
    name: `Zone ${config.camera_id}`,
    type: 'Restricted Zone',
    points: normalizedToPoints(config.polygon),
    color: ZONE_TYPE_COLORS['Restricted Zone'],
    enabled: true,
    createdAt: new Date().toISOString(),
  }
}

export function ZoneProvider({ children }: { children: ReactNode }) {
  const { cameras } = useCameras()
  const [zones, setZones] = useState<Zone[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const loadVersionRef = useRef(0)

  const reloadZones = useCallback(async (cameraIds: number[]) => {
    const version = ++loadVersionRef.current
    setLoading(true)
    try {
      const configs = await Promise.all(cameraIds.map((numId) => getZone(numId)))
      if (version !== loadVersionRef.current) return

      const loaded = configs.map(configToZone).filter((zone): zone is Zone => zone !== null)
      setZones(loaded)
      setError(null)
    } catch (err) {
      if (version !== loadVersionRef.current) return
      setError(err instanceof Error ? err.message : 'Failed to load zones')
    } finally {
      if (version === loadVersionRef.current) {
        setLoading(false)
      }
    }
  }, [])

  // Re-fetches whenever the set of registered cameras changes (add/remove) —
  // not on every CameraContext poll tick, since camera_id.join(',') is only
  // a new string when membership actually changes.
  const cameraIdsKey = cameras.map((c) => c.camera_id).join(',')
  useEffect(() => {
    const cameraIds = cameraIdsKey ? cameraIdsKey.split(',').map(Number) : []
    void reloadZones(cameraIds)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cameraIdsKey])

  const refreshZones = useCallback(
    () => reloadZones(cameras.map((c) => c.camera_id)),
    [reloadZones, cameras]
  )

  const getZonesByCamera = useCallback(
    (cameraId: string) => zones.filter((z) => z.cameraId === cameraId),
    [zones]
  )

  const saveCameraZone = useCallback(
    async (cameraId: string, points: Point[], alertClasses?: string[]) => {
      loadVersionRef.current += 1
      const numId = cameraNumId(cameraId)
      const config: ZoneConfig = {
        camera_id: numId,
        polygon: pointsToNormalized(points),
        alert_classes: alertClasses ?? [
          'person',
          'vehicle',
          'fire',
          'smoke',
          'handgun',
          'rifle',
          'pistol',
          'knife',
        ],
      }
      const saved = await apiUpdateZone(numId, config)
      const zone = configToZone(saved)
      setZones((prev) => {
        const withoutCamera = prev.filter((z) => z.cameraId !== cameraId)
        return zone ? [...withoutCamera, zone] : withoutCamera
      })
    },
    []
  )

  const deleteCameraZone = useCallback(async (cameraId: string) => {
    loadVersionRef.current += 1
    const numId = cameraNumId(cameraId)
    const saved = await apiUpdateZone(numId, {
      camera_id: numId,
      polygon: [],
      alert_classes: [],
    })
    const zone = configToZone(saved)
    setZones((prev) => {
      const withoutCamera = prev.filter((z) => z.cameraId !== cameraId)
      return zone ? [...withoutCamera, zone] : withoutCamera
    })
  }, [])

  return (
    <ZoneContext.Provider
      value={{
        zones,
        loading,
        error,
        getZonesByCamera,
        saveCameraZone,
        deleteCameraZone,
        reloadZones: refreshZones,
      }}
    >
      {children}
    </ZoneContext.Provider>
  )
}

export function useZones() {
  const ctx = useContext(ZoneContext)
  if (!ctx) throw new Error('useZones must be used within ZoneProvider')
  return ctx
}
