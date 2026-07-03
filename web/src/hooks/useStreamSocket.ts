import { useEffect, useRef, useState } from 'react'
import type { StreamFrame } from '@/types/api'

function wsUrl(cameraId: number): string {
  const base = import.meta.env.VITE_WS_BASE_URL
  if (base) return `${base}/ws/streams/${cameraId}`

  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${protocol}//${window.location.host}/ws/streams/${cameraId}`
}

export function useStreamSocket(cameraId: number) {
  const [frame, setFrame] = useState<StreamFrame | null>(null)
  const [connected, setConnected] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)
  const retryRef = useRef<number | null>(null)

  useEffect(() => {
    let cancelled = false

    function connect() {
      if (cancelled) return

      const ws = new WebSocket(wsUrl(cameraId))
      wsRef.current = ws

      ws.onopen = () => {
        if (!cancelled) setConnected(true)
      }

      ws.onmessage = (event) => {
        if (cancelled) return
        try {
          const payload = JSON.parse(event.data) as StreamFrame
          setFrame(payload)
        } catch {
          // ignore malformed frames
        }
      }

      ws.onclose = () => {
        if (cancelled) return
        setConnected(false)
        retryRef.current = window.setTimeout(connect, 2000)
      }

      ws.onerror = () => {
        ws.close()
      }
    }

    connect()

    return () => {
      cancelled = true
      if (retryRef.current !== null) window.clearTimeout(retryRef.current)
      wsRef.current?.close()
      wsRef.current = null
    }
  }, [cameraId])

  return { frame, connected }
}
