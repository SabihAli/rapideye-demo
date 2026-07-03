import { Pencil, Trash2 } from 'lucide-react'
import { Badge } from '@/components/ui/Badge'
import { Button } from '@/components/ui/Button'
import type { Zone } from '@/types/zone'

interface ZoneListProps {
  zones: Zone[]
  cameraName: string
  editingZoneId?: string | null
  onEdit: (zone: Zone) => void
  onDelete: (zone: Zone) => void
}

export function ZoneList({ zones, cameraName, editingZoneId, onEdit, onDelete }: ZoneListProps) {
  if (zones.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-white/10 py-8 text-center">
        <p className="text-sm text-muted">No zones on {cameraName}</p>
        <p className="mt-1 text-xs text-muted">Click Draw Zone to create one</p>
      </div>
    )
  }

  return (
    <div className="space-y-2">
      {zones.map((zone) => (
        <div
          key={zone.id}
          className={`rounded-xl border p-3 transition-colors ${
            editingZoneId === zone.id
              ? 'border-white/20 bg-bg-elevated'
              : 'border-white/5 bg-bg-secondary hover:border-white/10'
          }`}
        >
          <div className="flex items-start justify-between gap-2">
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <div className="h-3 w-3 shrink-0 rounded-sm" style={{ backgroundColor: zone.color }} />
                <p className="truncate text-sm font-medium text-text">{zone.name}</p>
              </div>
              <p className="mt-1 text-xs text-muted">{zone.type}</p>
              <p className="mt-0.5 text-xs text-muted">{zone.points.length} points</p>
            </div>
            <Badge variant={zone.enabled ? 'default' : 'outline'}>
              {zone.enabled ? 'Active' : 'Off'}
            </Badge>
          </div>
          <div className="mt-3 flex flex-wrap gap-1">
            <Button type="button" variant="outline" size="sm" onClick={() => onEdit(zone)}>
              <Pencil className="h-3.5 w-3.5" />
              Edit
            </Button>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={(e) => {
                e.stopPropagation()
                onDelete(zone)
              }}
            >
              <Trash2 className="h-3.5 w-3.5" />
              Remove
            </Button>
          </div>
        </div>
      ))}
    </div>
  )
}
