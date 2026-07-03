import { Maximize2, Cpu, Circle, Square, Disc } from 'lucide-react'
import { Link } from 'react-router-dom'
import { Card } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { Button } from '@/components/ui/Button'
import { CameraPreview } from '@/components/camera/CameraPreview'
import { ZoneOverlay } from '@/components/zones/ZoneOverlay'
import { useZones } from '@/context/ZoneContext'
import { useDemo } from '@/context/DemoContext'
import { useCameraRecording } from '@/hooks/useCameraRecording'
import { getZoneStates } from '@/lib/zoneUtils'
import type { CameraDefinition } from '@/lib/cameras'

interface CameraFeedCardProps {
  camera: CameraDefinition
}

function statusVariant(status: 'Online' | 'Offline' | 'Warning') {
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
  const { getStream, isStreamConnected, health } = useDemo()
  const { isRecording, loading: recordingLoading, start, stop } = useCameraRecording(camera.numId)
  const stream = getStream(camera.id)
  const zones = getZonesByCamera(camera.id)
  const { occupied, violated } = getZoneStates(zones, [])
  const connected = isStreamConnected(camera.id)
  const status = connected && stream ? 'Online' : health ? 'Warning' : 'Offline'
  const fps = stream?.target_fps ?? 0
  const latency = stream?.latency_ms ?? 0
  const isAlert = stream?.is_alert ?? false

  return (
    <Card
      className={`overflow-hidden transition-colors hover:border-white/10 ${isAlert ? 'ring-2 ring-danger/60' : ''}`}
    >
      <CameraPreview frameSrc={stream?.frame ?? null}>
        <ZoneOverlay zones={zones} occupiedIds={occupied} violatedIds={violated} showLabels={false} />

        <div className="absolute left-2 top-2 flex items-center gap-2">
          <Badge variant="danger" className="animate-pulse-live gap-1 bg-black/70">
            <Circle className="h-2 w-2 fill-danger" />
            LIVE
          </Badge>
          {isRecording && (
            <Badge variant="danger" className="gap-1 bg-black/70">
              <Disc className="h-2 w-2 fill-danger" />
              REC
            </Badge>
          )}
          {stream && (
            <span className="rounded bg-black/60 px-1.5 py-0.5 font-mono text-[10px] text-text">
              {latency.toFixed(0)}ms
            </span>
          )}
        </div>

        <div className="absolute right-2 top-2">
          <Badge variant={statusVariant(status)} className="bg-black/70">
            {status}
          </Badge>
        </div>

        {connected && (
          <div className="absolute bottom-2 left-2 flex items-center gap-1 rounded bg-black/60 px-2 py-1">
            <Cpu className="h-3 w-3 text-text-secondary" />
            <span className="text-[10px] text-text-secondary">AI Processing</span>
          </div>
        )}

        <div className="absolute bottom-2 right-2 font-mono text-[10px] text-muted">
          {fps.toFixed(1)} FPS
        </div>
      </CameraPreview>

      <div className="flex items-center justify-between gap-2 border-t border-white/5 p-3">
        <p className="truncate text-sm font-medium text-text">{camera.name}</p>
        <div className="flex items-center gap-2">
          <Button
            variant={isRecording ? 'danger' : 'outline'}
            size="sm"
            disabled={recordingLoading || !connected}
            onClick={() => void (isRecording ? stop() : start())}
          >
            {isRecording ? (
              <>
                <Square className="h-3.5 w-3.5" />
                <span className="hidden sm:inline">Stop</span>
              </>
            ) : (
              <>
                <Disc className="h-3.5 w-3.5" />
                <span className="hidden sm:inline">Record</span>
              </>
            )}
          </Button>
          <Link to={`/zones?camera=${camera.id}`}>
            <Button variant="outline" size="sm">
              <Maximize2 className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">Zones</span>
            </Button>
          </Link>
        </div>
      </div>
    </Card>
  )
}
