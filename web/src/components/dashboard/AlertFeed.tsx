import { ChevronRight, Play } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { recordingUrl } from '@/lib/api'
import { cameraFromNumId } from '@/lib/cameras'
import { formatAlertType, formatTimestamp } from '@/context/DemoContext'
import type { AlertEvent } from '@/types/api'

interface AlertFeedProps {
  alerts: AlertEvent[]
}

export function AlertFeed({ alerts }: AlertFeedProps) {
  return (
    <Card className="flex h-full flex-col">
      <CardHeader className="border-b border-white/5 pb-4">
        <div>
          <CardTitle className="text-base">Live Alert Feed</CardTitle>
          <p className="mt-0.5 text-xs text-muted">Real-time AI detections & events</p>
        </div>
      </CardHeader>

      <CardContent className="flex-1 space-y-2 overflow-y-auto p-3">
        {alerts.length === 0 ? (
          <div className="rounded-xl border border-dashed border-white/10 py-8 text-center text-sm text-muted">
            No alerts yet. Detections will appear here
          </div>
        ) : (
          alerts.map((alert) => {
            const camera = cameraFromNumId(alert.camera_id)
            const topDetection = alert.detections[0]
            const confidence = topDetection ? Math.round(topDetection.confidence * 100) : null

            return (
              <div
                key={alert.id}
                className="rounded-xl border border-white/5 bg-bg-secondary p-3 transition-colors hover:border-white/10"
              >
                <p className="text-sm font-medium text-text">{formatAlertType(alert.alert_type)}</p>
                <p className="mt-1 text-xs text-muted">
                  {topDetection
                    ? `${topDetection.class_name}${confidence !== null ? ` (${confidence}%)` : ''}`
                    : 'Detection event'}
                </p>
                <div className="mt-2 flex flex-wrap gap-x-2 gap-y-1 text-xs text-muted">
                  <span>{camera.name}</span>
                  <span>·</span>
                  <span>{formatTimestamp(alert.timestamp)}</span>
                </div>
                {alert.clip_path && (
                  <a
                    href={recordingUrl(alert.clip_path)}
                    target="_blank"
                    rel="noreferrer"
                    className="mt-2 inline-flex items-center gap-1 text-xs text-primary hover:underline"
                  >
                    <Play className="h-3 w-3" />
                    Play clip
                  </a>
                )}
              </div>
            )
          })
        )}
      </CardContent>

      <div className="border-t border-white/5 p-3">
        <Button variant="ghost" size="sm" className="w-full" disabled>
          View All Alerts
          <ChevronRight className="h-4 w-4" />
        </Button>
      </div>
    </Card>
  )
}
