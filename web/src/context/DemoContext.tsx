import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react'
import { getAlerts, getHealth } from '@/lib/api'
import { useStreamSocket } from '@/hooks/useStreamSocket'
import { useCameras } from '@/context/CameraContext'
import type { AlertEvent, StreamFrame, SystemHealth } from '@/types/api'

interface DemoContextValue {
  health: SystemHealth | null
  healthError: string | null
  alerts: AlertEvent[]
  refreshAlerts: () => Promise<void>
  getStream: (cameraId: string) => StreamFrame | null
  isStreamConnected: (cameraId: string) => boolean
}

const DemoContext = createContext<DemoContextValue | null>(null)

function useCameraStream(numId: number, enabled: boolean) {
  return useStreamSocket(numId, enabled)
}

function DemoStreamsProvider({
  children,
  registeredIds,
  onStreamsChange,
}: {
  children: ReactNode
  registeredIds: Set<number>
  onStreamsChange: (streams: Map<string, StreamFrame | null>, connected: Map<string, boolean>) => void
}) {
  const s1 = useCameraStream(1, registeredIds.has(1))
  const s2 = useCameraStream(2, registeredIds.has(2))
  const s3 = useCameraStream(3, registeredIds.has(3))
  const s4 = useCameraStream(4, registeredIds.has(4))

  useEffect(() => {
    const streams = new Map<string, StreamFrame | null>([
      ['cam-1', s1.frame],
      ['cam-2', s2.frame],
      ['cam-3', s3.frame],
      ['cam-4', s4.frame],
    ])
    const connected = new Map<string, boolean>([
      ['cam-1', s1.connected],
      ['cam-2', s2.connected],
      ['cam-3', s3.connected],
      ['cam-4', s4.connected],
    ])
    onStreamsChange(streams, connected)
  }, [s1.frame, s1.connected, s2.frame, s2.connected, s3.frame, s3.connected, s4.frame, s4.connected, onStreamsChange])

  return children
}

export function DemoProvider({ children }: { children: ReactNode }) {
  const { cameras } = useCameras()
  const [health, setHealth] = useState<SystemHealth | null>(null)
  const [healthError, setHealthError] = useState<string | null>(null)
  const [alerts, setAlerts] = useState<AlertEvent[]>([])
  const [streams, setStreams] = useState<Map<string, StreamFrame | null>>(new Map())
  const [connected, setConnected] = useState<Map<string, boolean>>(new Map())

  const registeredIds = useMemo(() => new Set(cameras.map((c) => c.camera_id)), [cameras])

  const onStreamsChange = useCallback(
    (nextStreams: Map<string, StreamFrame | null>, nextConnected: Map<string, boolean>) => {
      setStreams(nextStreams)
      setConnected(nextConnected)
    },
    []
  )

  const refreshHealth = useCallback(async () => {
    try {
      const data = await getHealth()
      setHealth(data)
      setHealthError(null)
    } catch (error) {
      setHealthError(error instanceof Error ? error.message : 'Backend unreachable')
    }
  }, [])

  const refreshAlerts = useCallback(async () => {
    try {
      const data = await getAlerts()
      setAlerts(data)
    } catch {
      // alerts panel stays on last known data
    }
  }, [])

  useEffect(() => {
    refreshHealth()
    refreshAlerts()
    const healthTimer = window.setInterval(refreshHealth, 5000)
    const alertsTimer = window.setInterval(refreshAlerts, 3000)
    return () => {
      window.clearInterval(healthTimer)
      window.clearInterval(alertsTimer)
    }
  }, [refreshHealth, refreshAlerts])

  const value = useMemo<DemoContextValue>(
    () => ({
      health,
      healthError,
      alerts,
      refreshAlerts,
      getStream: (cameraId: string) => streams.get(cameraId) ?? null,
      isStreamConnected: (cameraId: string) => connected.get(cameraId) ?? false,
    }),
    [health, healthError, alerts, refreshAlerts, streams, connected]
  )

  return (
    <DemoContext.Provider value={value}>
      <DemoStreamsProvider registeredIds={registeredIds} onStreamsChange={onStreamsChange}>
        {children}
      </DemoStreamsProvider>
    </DemoContext.Provider>
  )
}

export function useDemo() {
  const ctx = useContext(DemoContext)
  if (!ctx) throw new Error('useDemo must be used within DemoProvider')
  return ctx
}

export function formatAlertType(alertType: string): string {
  switch (alertType) {
    case 'fire.detected':
      return 'Fire Detected'
    case 'weapon.detected':
      return 'Weapon Detected'
    case 'zone.intrusion':
      return 'Zone Intrusion'
    default:
      return alertType.replace('.', ' ').replace(/\b\w/g, (c) => c.toUpperCase())
  }
}

export function formatTimestamp(epochSeconds: number): string {
  const date = new Date(epochSeconds * 1000)
  const now = Date.now()
  const diffMs = now - date.getTime()
  if (diffMs < 60_000) return 'Just now'
  if (diffMs < 3_600_000) return `${Math.floor(diffMs / 60_000)} min ago`
  return date.toLocaleTimeString()
}
