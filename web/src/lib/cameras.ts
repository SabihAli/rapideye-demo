import type { CameraOut } from '@/types/api'

export interface CameraDefinition {
  id: string
  numId: number
  name: string
}

export function toCameraDefinition(camera: CameraOut): CameraDefinition {
  return { id: `cam-${camera.camera_id}`, numId: camera.camera_id, name: camera.name }
}

// Live-updated by CameraProvider (see context/CameraContext.tsx) whenever the
// registered camera list changes. Components that need the reactive list for
// rendering should use useCameras() instead — this module-level store only
// backs the synchronous lookup helpers below, used outside render (e.g. in
// ZoneContext's data-mapping callbacks).
let cameraDefs: CameraDefinition[] = []

export function setCameraDefinitions(next: CameraDefinition[]): void {
  cameraDefs = next
}

export function cameraNumId(cameraId: string): number {
  const camera = cameraDefs.find((c) => c.id === cameraId)
  if (!camera) throw new Error(`Unknown camera id: ${cameraId}`)
  return camera.numId
}

export function cameraFromNumId(numId: number): CameraDefinition {
  const camera = cameraDefs.find((c) => c.numId === numId)
  // Falls back instead of throwing: alerts/recordings can reference a
  // camera_id that has since been removed, and that historical data still
  // needs to render.
  return camera ?? { id: `cam-${numId}`, numId, name: `Camera ${numId} (removed)` }
}
