import { useEffect, useState } from 'react'
import { Maximize2, Cpu, Circle, Square } from 'lucide-react'
import { Link } from 'react-router-dom'
import { Card } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { Button } from '@/components/ui/Button'
import { CameraPreview } from '@/components/camera/CameraPreview'
import { ZoneOverlay } from '@/components/zones/ZoneOverlay'
import { useZones } from '@/context/ZoneContext'
import { useDemo } from '@/context/DemoContext'
import { getZoneStates } from '@/lib/zoneUtils'
import { cn } from '@/lib/utils'
import type { CameraDefinition } from '@/lib/cameras'

interface CameraFeedCardProps {
  camera: CameraDefinition
}

type DetectionMode = 'fire' | 'face' | 'weapon'

const DETECTION_OPTIONS: { id: DetectionMode; label: string }[] = [
  { id: 'fire', label: 'Fire' },
  { id: 'face', label: 'Face Recognition' },
  { id: 'weapon', label: 'Weapon' },
]

function formatRecordingTime(totalSeconds: number): string {
  const hours = Math.floor(totalSeconds / 3600)
  const minutes = Math.floor((totalSeconds % 3600) / 60)
  const seconds = totalSeconds % 60
  const mm = minutes.toString().padStart(2, '0')
  const ss = seconds.toString().padStart(2, '0')
  return hours > 0 ? `${hours}:${mm}:${ss}` : `${mm}:${ss}`
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
  const [isRecording, setIsRecording] = useState(false)
  const [recordingSeconds, setRecordingSeconds] = useState(0)
  const [selectedDetections, setSelectedDetections] = useState<Record<DetectionMode, boolean>>({
    fire: false,
    face: false,
    weapon: false,
  })
  const stream = getStream(camera.id)
  const zones = getZonesByCamera(camera.id)
  const { occupied, violated } = getZoneStates(zones, [])
  const connected = isStreamConnected(camera.id)
  const status = connected && stream ? 'Online' : health ? 'Warning' : 'Offline'
  const fps = stream?.target_fps ?? 0
  const latency = stream?.latency_ms ?? 0
  const isAlert = stream?.is_alert ?? false

  useEffect(() => {
    if (!isRecording) return
    const timer = window.setInterval(() => {
      setRecordingSeconds((prev) => prev + 1)
    }, 1000)
    return () => window.clearInterval(timer)
  }, [isRecording])

  const handleRecordingToggle = () => {
    if (isRecording) {
      setIsRecording(false)
      return
    }
    setRecordingSeconds(0)
    setIsRecording(true)
  }

  const toggleDetection = (id: DetectionMode) => {
    setSelectedDetections((prev) => ({ ...prev, [id]: !prev[id] }))
  }

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
            <Badge variant="danger" className="animate-pulse-live gap-1 bg-black/70 font-mono tabular-nums">
              <Circle className="h-2 w-2 fill-danger" />
              {formatRecordingTime(recordingSeconds)}
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

      <div className="border-t border-white/5 p-3">
        <div className="flex items-center justify-between gap-2">
          <p className="truncate text-sm font-medium text-text">{camera.name}</p>
          <div className="flex shrink-0 items-center gap-2">
            <Button
              variant={isRecording ? 'danger' : 'outline'}
              size="sm"
              onClick={handleRecordingToggle}
              aria-pressed={isRecording}
              aria-label={isRecording ? 'Stop recording' : 'Start recording'}
            >
              {isRecording ? (
                <>
                  <Square className="h-3 w-3 fill-current" />
                  Stop
                </>
              ) : (
                <>
                  <Circle className="h-3 w-3 fill-danger text-danger" />
                  Record
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
        <fieldset className="mt-2.5 border-0 border-t border-white/5 p-0 pt-2.5">
          <legend className="sr-only">Detection modes for {camera.name}</legend>
          <div className="grid grid-cols-3 gap-2">
            {DETECTION_OPTIONS.map(({ id, label }) => {
              const selected = selectedDetections[id]
              return (
                <label
                  key={id}
                  className={cn(
                    'flex cursor-pointer flex-col items-center gap-1.5 rounded-lg border px-1.5 py-2 text-center transition-colors',
                    selected
                      ? 'border-white/25 bg-bg-secondary text-text'
                      : 'border-white/10 text-muted hover:border-white/20 hover:text-text-secondary'
                  )}
                >
                  <input
                    type="checkbox"
                    name={`detection-${camera.id}-${id}`}
                    checked={selected}
                    onChange={() => toggleDetection(id)}
                    className="h-3.5 w-3.5 shrink-0 rounded accent-white"
                  />
                  <span className="text-[10px] leading-tight">{label}</span>
                </label>
              )
            })}
          </div>
        </fieldset>
      </div>
    </Card>
  )
}
