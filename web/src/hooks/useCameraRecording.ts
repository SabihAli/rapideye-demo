import { useCallback, useEffect, useState } from 'react'
import {
  getCameraRecordingStatus,
  startCameraRecording,
  stopCameraRecording,
} from '@/lib/api'
import type { CameraRecordingStatus } from '@/types/api'

export function useCameraRecording(cameraId: number) {
  const [status, setStatus] = useState<CameraRecordingStatus | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    try {
      const next = await getCameraRecordingStatus(cameraId)
      setStatus(next)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load recording status')
    }
  }, [cameraId])

  useEffect(() => {
    void refresh()
    const interval = window.setInterval(() => {
      void refresh()
    }, 3000)
    return () => window.clearInterval(interval)
  }, [refresh])

  const start = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const recording = await startCameraRecording(cameraId)
      setStatus({
        camera_id: cameraId,
        is_recording: true,
        recording,
      })
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start recording')
    } finally {
      setLoading(false)
    }
  }, [cameraId])

  const stop = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const recording = await stopCameraRecording(cameraId)
      setStatus({
        camera_id: cameraId,
        is_recording: false,
        recording,
      })
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to stop recording')
    } finally {
      setLoading(false)
    }
  }, [cameraId])

  return {
    isRecording: status?.is_recording ?? false,
    recording: status?.recording ?? null,
    loading,
    error,
    start,
    stop,
    refresh,
  }
}
