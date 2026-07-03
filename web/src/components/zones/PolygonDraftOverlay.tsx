import type { Point } from '@/types/zone'
import { pointsToSvg } from '@/lib/zoneUtils'

interface PolygonDraftOverlayProps {
  points: Point[]
  closed?: boolean
  cursorPoint?: Point | null
}

/** SVG lines only — point markers are rendered as HTML to avoid stretch/distortion */
export function PolygonDraftOverlay({ points, closed = false, cursorPoint }: PolygonDraftOverlayProps) {
  if (points.length === 0 && !cursorPoint) return null

  const linePoints = cursorPoint && !closed && points.length > 0 ? [...points, cursorPoint] : points

  return (
    <svg
      className="pointer-events-none absolute inset-0 h-full w-full"
      viewBox="0 0 100 100"
      preserveAspectRatio="none"
    >
      {!closed && linePoints.length >= 2 && (
        <polyline
          points={pointsToSvg(linePoints)}
          fill="none"
          stroke="rgba(255,255,255,0.7)"
          strokeWidth="0.5"
          vectorEffect="non-scaling-stroke"
          strokeDasharray="4,3"
        />
      )}
      {closed && points.length >= 3 && (
        <polygon
          points={pointsToSvg(points)}
          fill="rgba(255,255,255,0.08)"
          stroke="rgba(255,255,255,0.7)"
          strokeWidth="0.6"
          vectorEffect="non-scaling-stroke"
        />
      )}
    </svg>
  )
}
