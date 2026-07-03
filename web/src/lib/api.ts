import type { AlertEvent, StreamStatus, SystemHealth, ZoneConfig } from '@/types/api'

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

export function startDemo(): Promise<{ status: string; message: string }> {
  return request('/api/demo/start', { method: 'POST' })
}

export function recordingUrl(clipPath: string): string {
  if (clipPath.startsWith('http')) return clipPath
  return `${API_BASE}${clipPath}`
}
