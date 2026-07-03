import type { Point } from '@/types/zone'
import { cn } from '@/lib/utils'

interface DraftPointMarkersProps {
  points: Point[]
  selectedIndex?: number | null
  onSelect?: (index: number | null) => void
  interactive?: boolean
}

export function DraftPointMarkers({
  points,
  selectedIndex = null,
  onSelect,
  interactive = false,
}: DraftPointMarkersProps) {
  if (points.length === 0) return null

  return (
    <div className="pointer-events-none absolute inset-0">
      {points.map((pt, i) => {
        const selected = selectedIndex === i
        return (
          <button
            key={`pt-${i}-${pt.x.toFixed(1)}-${pt.y.toFixed(1)}`}
            type="button"
            disabled={!interactive}
            onClick={(e) => {
              if (!interactive || !onSelect) return
              e.stopPropagation()
              onSelect(selected ? null : i)
            }}
            className={cn(
              'absolute flex -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full border-2 font-bold shadow-md transition-transform',
              interactive ? 'pointer-events-auto cursor-pointer' : 'pointer-events-none',
              selected
                ? 'h-7 w-7 border-white bg-white text-inverse text-xs scale-110'
                : 'h-6 w-6 border-black/80 bg-white/90 text-inverse text-[11px]'
            )}
            style={{ left: `${pt.x}%`, top: `${pt.y}%` }}
            title={interactive ? `Point ${i + 1}` : `Point ${i + 1}`}
          >
            {i + 1}
          </button>
        )
      })}
    </div>
  )
}
