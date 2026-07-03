import type { Detection } from '@/data/mockData'
import type { Point, Zone, ZoneType } from '@/types/zone'

export function pointsToSvg(points: Point[]): string {
  return points.map((p) => `${p.x},${p.y}`).join(' ')
}

export function rectToPoints(x1: number, y1: number, x2: number, y2: number): Point[] {
  const left = Math.min(x1, x2)
  const right = Math.max(x1, x2)
  const top = Math.min(y1, y2)
  const bottom = Math.max(y1, y2)
  return [
    { x: left, y: top },
    { x: right, y: top },
    { x: right, y: bottom },
    { x: left, y: bottom },
  ]
}

export function toPercent(clientX: number, clientY: number, rect: DOMRect): Point {
  return {
    x: Math.min(100, Math.max(0, ((clientX - rect.left) / rect.width) * 100)),
    y: Math.min(100, Math.max(0, ((clientY - rect.top) / rect.height) * 100)),
  }
}

export function detectionCenter(det: Detection): Point {
  return { x: det.x + det.width / 2, y: det.y + det.height / 2 }
}

export function isInsidePolygon(point: Point, polygon: Point[]): boolean {
  let inside = false
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i].x
    const yi = polygon[i].y
    const xj = polygon[j].x
    const yj = polygon[j].y
    const intersect =
      yi > point.y !== yj > point.y &&
      point.x < ((xj - xi) * (point.y - yi)) / (yj - yi) + xi
    if (intersect) inside = !inside
  }
  return inside
}

export interface ZoneViolation {
  zoneId: string
  detectionId: string
  eventType: string
}

function isViolation(zoneType: ZoneType, label: string): boolean {
  switch (zoneType) {
    case 'Restricted Zone':
    case 'No-Entry Area':
      return label === 'Unknown' || label === 'Person'
    case 'PPE Required Zone':
      return label === 'No Helmet'
    case 'Fire Risk Zone':
      return label === 'Fire' || label === 'Smoke'
    case 'Lab Area':
      return label === 'Unknown'
    default:
      return false
  }
}

export function getZoneStates(
  zones: Zone[],
  detections: Detection[]
): { occupied: Set<string>; violated: Set<string> } {
  const occupied = new Set<string>()
  const violated = new Set<string>()

  for (const det of detections) {
    const center = detectionCenter(det)
    for (const zone of zones) {
      if (!zone.enabled) continue
      if (!isInsidePolygon(center, zone.points)) continue

      occupied.add(zone.id)
      if (isViolation(zone.type, det.label)) {
        violated.add(zone.id)
      }
    }
  }

  return { occupied, violated }
}

export function getZoneViolations(zones: Zone[], detections: Detection[]): ZoneViolation[] {
  const violations: ZoneViolation[] = []

  for (const det of detections) {
    const center = detectionCenter(det)
    for (const zone of zones) {
      if (!zone.enabled) continue
      if (!isInsidePolygon(center, zone.points)) continue
      if (isViolation(zone.type, det.label)) {
        violations.push({
          zoneId: zone.id,
          detectionId: det.id,
          eventType: `${zone.type} violation`,
        })
      }
    }
  }

  return violations
}
