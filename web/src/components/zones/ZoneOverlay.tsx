import type { Zone } from '@/types/zone'
import { pointsToSvg } from '@/lib/zoneUtils'
import { cn } from '@/lib/utils'

interface ZoneOverlayProps {
  zones: Zone[]
  occupiedIds?: Set<string>
  violatedIds?: Set<string>
  showLabels?: boolean
  previewPoints?: { x: number; y: number }[] | null
}

function getZoneStyle(zone: Zone, occupied: boolean, violated: boolean) {
  if (!zone.enabled) {
    return { stroke: '#64748B', fill: 'rgba(100,116,139,0.1)', dash: '4,4', width: 0.4 }
  }
  if (violated) {
    return { stroke: '#EF4444', fill: 'rgba(239,68,68,0.25)', dash: '0', width: 0.8 }
  }
  if (occupied) {
    return { stroke: zone.color, fill: `${zone.color}30`, dash: '0', width: 0.6 }
  }
  return { stroke: zone.color, fill: `${zone.color}15`, dash: '4,2', width: 0.5 }
}

export function ZoneOverlay({
  zones,
  occupiedIds = new Set(),
  violatedIds = new Set(),
  showLabels = true,
  previewPoints,
}: ZoneOverlayProps) {
  return (
    <svg className="pointer-events-none absolute inset-0 h-full w-full" viewBox="0 0 100 100" preserveAspectRatio="none">
      {zones.map((zone) => {
        const occupied = occupiedIds.has(zone.id)
        const violated = violatedIds.has(zone.id)
        const style = getZoneStyle(zone, occupied, violated)

        return (
          <g key={zone.id}>
            <polygon
              points={pointsToSvg(zone.points)}
              fill={style.fill}
              stroke={style.stroke}
              strokeWidth={style.width}
              strokeDasharray={style.dash}
              className={cn(violated && 'animate-pulse-live')}
            />
            {showLabels && (
              <text
                x={zone.points[0].x + 1}
                y={zone.points[0].y + 3}
                fill={violated ? '#EF4444' : zone.color}
                fontSize="2.5"
                fontWeight="600"
              >
                {zone.name}
              </text>
            )}
          </g>
        )
      })}
      {previewPoints && previewPoints.length >= 3 && (
        <polygon
          points={pointsToSvg(previewPoints)}
          fill="rgba(56,189,248,0.15)"
          stroke="#38BDF8"
          strokeWidth="0.6"
          strokeDasharray="3,2"
        />
      )}
      {previewPoints && previewPoints.length >= 2 && previewPoints.length < 3 && (
        <polyline
          points={pointsToSvg(previewPoints)}
          fill="none"
          stroke="#38BDF8"
          strokeWidth="0.5"
          strokeDasharray="2,1"
        />
      )}
    </svg>
  )
}
