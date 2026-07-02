import { Maximize2, Cpu, Circle } from 'lucide-react'
import { Link } from 'react-router-dom'
import { Card } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { Button } from '@/components/ui/Button'
import { CameraPreview } from '@/components/camera/CameraPreview'
import { ZoneOverlay } from '@/components/zones/ZoneOverlay'
import { DetectionOverlay } from '@/components/zones/DetectionOverlay'
import { useZones } from '@/context/ZoneContext'
import { getZoneStates } from '@/lib/zoneUtils'
import type { CameraFeed } from '@/data/mockData'

interface CameraFeedCardProps {
  camera: CameraFeed
}

function statusVariant(status: CameraFeed['status']) {
  switch (status) {
    case 'Online':
      return 'success'
    case 'Warning':
      return 'warning'
    case 'Offline':
      return 'danger'
  }
}

export function CameraFeedCard({ camera }: CameraFeedCardProps) {
  const { getZonesByCamera } = useZones()
  const zones = getZonesByCamera(camera.id)
  const { occupied, violated } = getZoneStates(zones, camera.detections)

  return (
    <Card className="overflow-hidden transition-colors hover:border-white/10">
      <CameraPreview camera={camera}>
        <ZoneOverlay zones={zones} occupiedIds={occupied} violatedIds={violated} showLabels={false} />
        <DetectionOverlay detections={camera.detections} />

        <div className="absolute left-2 top-2 flex items-center gap-2">
          <Badge variant="danger" className="animate-pulse-live gap-1 bg-black/70">
            <Circle className="h-2 w-2 fill-danger" />
            LIVE
          </Badge>
          <span className="rounded bg-black/60 px-1.5 py-0.5 font-mono text-[10px] text-text">
            {camera.timestamp}
          </span>
        </div>

        <div className="absolute right-2 top-2">
          <Badge variant={statusVariant(camera.status)} className="bg-black/70">
            {camera.status}
          </Badge>
        </div>

        {camera.aiProcessing && (
          <div className="absolute bottom-2 left-2 flex items-center gap-1 rounded bg-black/60 px-2 py-1">
            <Cpu className="h-3 w-3 text-text-secondary" />
            <span className="text-[10px] text-text-secondary">AI Processing</span>
          </div>
        )}

        <div className="absolute bottom-2 right-2 font-mono text-[10px] text-muted">
          {camera.fps} FPS
        </div>
      </CameraPreview>

      <div className="flex items-center justify-between border-t border-white/5 p-3">
        <p className="truncate text-sm font-medium text-text">{camera.name}</p>
        <Link to={`/zones?camera=${camera.id}`}>
          <Button variant="outline" size="sm">
            <Maximize2 className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Zones</span>
          </Button>
        </Link>
      </div>
    </Card>
  )
}
