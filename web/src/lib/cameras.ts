export interface CameraDefinition {
  id: string
  numId: number
  name: string
}

export const CAMERAS: CameraDefinition[] = [
  { id: 'cam-1', numId: 1, name: 'Camera 1' },
  { id: 'cam-2', numId: 2, name: 'Camera 2' },
  { id: 'cam-3', numId: 3, name: 'Camera 3' },
  { id: 'cam-4', numId: 4, name: 'Camera 4' },
]

export function cameraNumId(cameraId: string): number {
  const camera = CAMERAS.find((c) => c.id === cameraId)
  if (!camera) throw new Error(`Unknown camera id: ${cameraId}`)
  return camera.numId
}

export function cameraFromNumId(numId: number): CameraDefinition {
  const camera = CAMERAS.find((c) => c.numId === numId)
  if (!camera) throw new Error(`Unknown camera numId: ${numId}`)
  return camera
}
