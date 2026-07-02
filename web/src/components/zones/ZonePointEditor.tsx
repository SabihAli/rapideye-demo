import { Trash2, Undo2 } from 'lucide-react'
import { Button } from '@/components/ui/Button'
import type { Point } from '@/types/zone'
import { MAX_ZONE_POINTS, MIN_ZONE_POINTS } from '@/types/zone'
import { cn } from '@/lib/utils'

interface ZonePointEditorProps {
  points: Point[]
  selectedIndex?: number | null
  onSelectPoint?: (index: number | null) => void
  onChange: (points: Point[]) => void
  onUndo: () => void
  onClear: () => void
}

export function ZonePointEditor({
  points,
  selectedIndex = null,
  onSelectPoint,
  onChange,
  onUndo,
  onClear,
}: ZonePointEditorProps) {
  const updatePoint = (index: number, axis: 'x' | 'y', value: number) => {
    const clamped = Math.min(100, Math.max(0, value))
    const next = points.map((p, i) => (i === index ? { ...p, [axis]: clamped } : p))
    onChange(next)
  }

  const removePoint = (index: number) => {
    onChange(points.filter((_, i) => i !== index))
    onSelectPoint?.(null)
  }

  return (
    <div className="space-y-3 border-t border-white/5 pt-4">
      <div className="flex items-center justify-between">
        <p className="text-sm font-medium text-text">
          Points ({points.length}/{MAX_ZONE_POINTS})
        </p>
        <div className="flex gap-1">
          <Button variant="ghost" size="sm" onClick={onUndo} disabled={points.length === 0}>
            <Undo2 className="h-3.5 w-3.5" />
            Undo
          </Button>
          <Button variant="ghost" size="sm" onClick={onClear} disabled={points.length === 0}>
            Clear
          </Button>
        </div>
      </div>

      <p className="text-xs text-muted">
        Click the camera feed to place points (minimum {MIN_ZONE_POINTS}).
      </p>

      {points.length === 0 ? (
        <div className="rounded-xl border border-dashed border-white/10 py-4 text-center text-xs text-muted">
          No points yet — click on the camera view to start
        </div>
      ) : (
        <div className="max-h-40 space-y-1.5 overflow-y-auto">
          {points.map((pt, i) => (
            <div
              key={`row-${i}`}
              onClick={() => onSelectPoint?.(selectedIndex === i ? null : i)}
              className={cn(
                'flex cursor-pointer items-center gap-2 rounded-lg border px-2 py-1.5 transition-colors',
                selectedIndex === i
                  ? 'border-white/20 bg-bg-elevated'
                  : 'border-white/5 bg-bg-secondary hover:border-white/10'
              )}
            >
              <span className="w-14 shrink-0 text-xs font-medium text-text-secondary">
                Point {i + 1}
              </span>
              <label className="flex items-center gap-1 text-xs text-muted" onClick={(e) => e.stopPropagation()}>
                X
                <input
                  type="number"
                  min={0}
                  max={100}
                  step={0.1}
                  value={pt.x}
                  onChange={(e) => updatePoint(i, 'x', parseFloat(e.target.value) || 0)}
                  className="w-16 rounded-lg border border-white/10 bg-bg-main px-1.5 py-1 text-xs text-text focus:border-white/25 focus:outline-none"
                />
              </label>
              <label className="flex items-center gap-1 text-xs text-muted" onClick={(e) => e.stopPropagation()}>
                Y
                <input
                  type="number"
                  min={0}
                  max={100}
                  step={0.1}
                  value={pt.y}
                  onChange={(e) => updatePoint(i, 'y', parseFloat(e.target.value) || 0)}
                  className="w-16 rounded-lg border border-white/10 bg-bg-main px-1.5 py-1 text-xs text-text focus:border-white/25 focus:outline-none"
                />
              </label>
              <button
                type="button"
                onClick={(e) => { e.stopPropagation(); removePoint(i) }}
                className="ml-auto rounded p-1 text-muted hover:bg-bg-elevated hover:text-danger"
                title="Remove point"
              >
                <Trash2 className="h-3.5 w-3.5" />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
