import { createContext, useContext, useState, useCallback, type ReactNode } from 'react'
import { initialZones } from '@/data/mockData'
import type { Zone } from '@/types/zone'
import { ZONE_TYPE_COLORS } from '@/types/zone'

interface ZoneContextValue {
  zones: Zone[]
  getZonesByCamera: (cameraId: string) => Zone[]
  addZone: (zone: Omit<Zone, 'id' | 'createdAt'>) => void
  updateZone: (id: string, data: Partial<Zone>) => void
  deleteZone: (id: string) => void
  toggleZone: (id: string) => void
}

const ZoneContext = createContext<ZoneContextValue | null>(null)

export function ZoneProvider({ children }: { children: ReactNode }) {
  const [zones, setZones] = useState<Zone[]>(initialZones)

  const getZonesByCamera = useCallback(
    (cameraId: string) => zones.filter((z) => z.cameraId === cameraId),
    [zones]
  )

  const addZone = useCallback((zone: Omit<Zone, 'id' | 'createdAt'>) => {
    const newZone: Zone = {
      ...zone,
      id: `zone-${Date.now()}`,
      color: zone.color || ZONE_TYPE_COLORS[zone.type],
      createdAt: new Date().toISOString(),
    }
    setZones((prev) => [...prev, newZone])
  }, [])

  const updateZone = useCallback((id: string, data: Partial<Zone>) => {
    setZones((prev) => prev.map((z) => (z.id === id ? { ...z, ...data } : z)))
  }, [])

  const deleteZone = useCallback((id: string) => {
    setZones((prev) => prev.filter((z) => z.id !== id))
  }, [])

  const toggleZone = useCallback((id: string) => {
    setZones((prev) =>
      prev.map((z) => (z.id === id ? { ...z, enabled: !z.enabled } : z))
    )
  }, [])

  return (
    <ZoneContext.Provider
      value={{ zones, getZonesByCamera, addZone, updateZone, deleteZone, toggleZone }}
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
