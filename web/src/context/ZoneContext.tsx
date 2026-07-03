import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from 'react'
import { getZone, updateZone as apiUpdateZone } from '@/lib/api'
import { CAMERAS, cameraNumId } from '@/lib/cameras'
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

  const camera = CAMERAS.find((c) => c.numId === config.camera_id)
  if (!camera) return null

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
  const [zones, setZones] = useState<Zone[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const reloadZones = useCallback(async () => {
    setLoading(true)
    try {
      const configs = await Promise.all(CAMERAS.map((camera) => getZone(camera.numId)))
      const loaded = configs.map(configToZone).filter((zone): zone is Zone => zone !== null)
      setZones(loaded)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load zones')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    reloadZones()
  }, [reloadZones])

  const getZonesByCamera = useCallback(
    (cameraId: string) => zones.filter((z) => z.cameraId === cameraId),
    [zones]
  )

  const saveCameraZone = useCallback(
    async (cameraId: string, points: Point[], alertClasses?: string[]) => {
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
    const numId = cameraNumId(cameraId)
    const config: ZoneConfig = {
      camera_id: numId,
      polygon: [],
      alert_classes: [],
    }
    await apiUpdateZone(numId, config)
    setZones((prev) => prev.filter((z) => z.cameraId !== cameraId))
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
        reloadZones,
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
