import { useMemo } from 'react'
import { Film } from 'lucide-react'
import { Card, CardContent } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { recordingUrl } from '@/lib/api'
import { cameraFromNumId } from '@/lib/cameras'
import { formatAlertType, formatTimestamp, useDemo } from '@/context/DemoContext'

export function RecordingsPage() {
  const { alerts } = useDemo()

  const recordings = useMemo(
    () => alerts.filter((alert) => alert.clip_path).sort((a, b) => b.timestamp - a.timestamp),
    [alerts]
  )

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-bold text-text">Recordings</h1>
        <p className="text-sm text-muted">Alert clips captured from camera streams</p>
      </div>

      {recordings.length === 0 ? (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-16 text-center">
            <Film className="h-12 w-12 text-muted" />
            <p className="mt-4 text-sm font-medium text-text">No recordings yet</p>
            <p className="mt-1 max-w-sm text-xs text-muted">
              Recorded clips appear here when the system captures alert events from your cameras.
            </p>
          </CardContent>
        </Card>
      ) : (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
          {recordings.map((alert) => {
            const camera = cameraFromNumId(alert.camera_id)
            const topDetection = alert.detections[0]
            const confidence = topDetection ? Math.round(topDetection.confidence * 100) : null

            return (
              <Card key={alert.id} className="overflow-hidden">
                <div className="aspect-video bg-black">
                  <video
                    src={recordingUrl(alert.clip_path!)}
                    controls
                    preload="metadata"
                    className="h-full w-full object-contain"
                  />
                </div>
                <CardContent className="space-y-2 p-4">
                  <div className="flex items-start justify-between gap-2">
                    <p className="text-sm font-medium text-text">{formatAlertType(alert.alert_type)}</p>
                    <Badge variant="info">{camera.name}</Badge>
                  </div>
                  {topDetection && (
                    <p className="text-xs text-muted">
                      {topDetection.class_name}
                      {confidence !== null ? ` (${confidence}%)` : ''}
                    </p>
                  )}
                  <p className="text-xs text-muted">{formatTimestamp(alert.timestamp)}</p>
                </CardContent>
              </Card>
            )
          })}
        </div>
      )}
    </div>
  )
}
