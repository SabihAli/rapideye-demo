import type {
  AlertEvent,
  CameraRecording,
  CameraRecordingStatus,
  StreamStatus,
  SystemHealth,
  ZoneConfig,
} from '@/types/api'

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? ''

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...init?.headers },
    ...init,
  })

  if (!response.ok) {
    const detail = await response.text()
    throw new Error(detail || `Request failed: ${response.status}`)
  }

  return response.json() as Promise<T>
}

export function getHealth(): Promise<SystemHealth> {
  return request('/api/health')
}

export function getStreamsStatus(): Promise<StreamStatus[]> {
  return request('/api/streams/status')
}

export function getAlerts(limit = 50, offset = 0): Promise<AlertEvent[]> {
  return request(`/api/alerts?limit=${limit}&offset=${offset}`)
}

export function getZone(cameraId: number): Promise<ZoneConfig> {
  return request(`/api/zones/${cameraId}`)
}

export function updateZone(cameraId: number, config: ZoneConfig): Promise<ZoneConfig> {
  return request(`/api/zones/${cameraId}`, {
    method: 'PUT',
    body: JSON.stringify(config),
  })
}

export function recordingUrl(clipPath: string): string {
  if (clipPath.startsWith('http')) return clipPath
  return `${API_BASE}${clipPath}`
}

export function getCameraRecordingStatus(cameraId: number): Promise<CameraRecordingStatus> {
  return request(`/api/cameras/${cameraId}/recordings/status`)
}

export function startCameraRecording(cameraId: number): Promise<CameraRecording> {
  return request(`/api/cameras/${cameraId}/recordings/start`, { method: 'POST' })
}

export function stopCameraRecording(cameraId: number): Promise<CameraRecording> {
  return request(`/api/cameras/${cameraId}/recordings/stop`, { method: 'POST' })
}

export function getCameraRecordings(
  cameraId: number,
  limit = 50,
  offset = 0
): Promise<CameraRecording[]> {
  return request(`/api/cameras/${cameraId}/recordings?limit=${limit}&offset=${offset}`)
}

export function getAllCameraRecordings(limit = 50, offset = 0): Promise<CameraRecording[]> {
  return request(`/api/camera-recordings?limit=${limit}&offset=${offset}`)
}
