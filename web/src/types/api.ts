export interface ApiDetection {
  bbox: [number, number, number, number]
  class_name: string
  confidence: number
}

export interface StreamFrame {
  camera_id: number
  frame: string
  detections: ApiDetection[]
  is_alert: boolean
  target_fps: number
  latency_ms: number
}

export interface StreamStatus {
  camera_id: number
  is_active: boolean
  current_fps: number
  error_count: number
}

export interface SystemHealth {
  gpu_available: boolean
  gpu_device?: string | null
  models_loaded: string[]
  streams: StreamStatus[]
}

export interface ZoneConfig {
  camera_id: number
  polygon: [number, number][]
  alert_classes: string[]
}

export interface AlertEvent {
  id: string
  camera_id: number
  alert_type: string
  timestamp: number
  detections: ApiDetection[]
  clip_path?: string | null
}
