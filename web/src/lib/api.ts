import type {
  AlertEvent,
  CameraModelSwitches,
  CameraOut,
  CameraRecording,
  CameraRecordingStatus,
  StreamStatus,
  SystemHealth,
  ZoneConfig,
} from '@/types/api'

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? ''

function parseErrorDetail(body: string, status: number): Error {
  try {
    const parsed = JSON.parse(body) as { detail?: string }
    if (parsed.detail) return new Error(parsed.detail)
  } catch {
    // body wasn't JSON — fall through to the raw-text/status message below
  }
  return new Error(body || `Request failed: ${status}`)
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...init?.headers },
    ...init,
  })

  if (!response.ok) {
    throw parseErrorDetail(await response.text(), response.status)
  }

  if (response.status === 204) return undefined as T

  return response.json() as Promise<T>
}

/** Uploads a FormData body via XHR (not fetch) so progress can be reported
 * during a large video-file upload — fetch has no upload-progress event. */
function requestFormWithProgress<T>(
  path: string,
  form: FormData,
  onProgress?: (pct: number) => void
): Promise<T> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', `${API_BASE}${path}`)

    xhr.upload.onprogress = (event) => {
      if (onProgress && event.lengthComputable) {
        onProgress(Math.round((event.loaded / event.total) * 100))
      }
    }

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText) as T)
        } catch {
          reject(new Error('Malformed response from server'))
        }
      } else {
        reject(parseErrorDetail(xhr.responseText, xhr.status))
      }
    }

    xhr.onerror = () => reject(new Error('Network error — is the backend reachable?'))

    xhr.send(form)
  })
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

export function getAllModelSwitches(): Promise<CameraModelSwitches[]> {
  return request('/api/cameras/model-switches')
}

export function getModelSwitches(cameraId: number): Promise<CameraModelSwitches> {
  return request(`/api/cameras/${cameraId}/model-switches`)
}

export function updateModelSwitches(config: CameraModelSwitches): Promise<CameraModelSwitches> {
  return request(`/api/cameras/${config.camera_id}/model-switches`, {
    method: 'PUT',
    body: JSON.stringify(config),
  })
}

export function listCameras(): Promise<CameraOut[]> {
  return request('/api/cameras')
}

export function addCameraByUrl(name: string, url: string): Promise<CameraOut> {
  const form = new FormData()
  form.append('name', name)
  form.append('source_type', 'url')
  form.append('url', url)
  return requestFormWithProgress('/api/cameras', form)
}

export function addCameraByFile(
  name: string,
  file: File,
  onProgress?: (pct: number) => void
): Promise<CameraOut> {
  const form = new FormData()
  form.append('name', name)
  form.append('source_type', 'file')
  form.append('file', file)
  return requestFormWithProgress('/api/cameras', form, onProgress)
}

export function removeCamera(cameraId: number): Promise<void> {
  return request(`/api/cameras/${cameraId}`, { method: 'DELETE' })
}
