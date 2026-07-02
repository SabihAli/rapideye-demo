import { ChevronRight } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import type { Alert } from '@/data/mockData'

interface AlertFeedProps {
  alerts: Alert[]
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
        {alerts.map((alert) => (
          <div
            key={alert.id}
            className="rounded-xl border border-white/5 bg-bg-secondary p-3 transition-colors hover:border-white/10"
          >
            <p className="text-sm font-medium text-text">{alert.title}</p>
            <p className="mt-1 text-xs text-muted">{alert.description}</p>
            <div className="mt-2 flex flex-wrap gap-x-2 gap-y-1 text-xs text-muted">
              <span>{alert.camera}</span>
              <span>·</span>
              <span>{alert.zone}</span>
              <span>·</span>
              <span>{alert.timestamp}</span>
            </div>
          </div>
        ))}
      </CardContent>

      <div className="border-t border-white/5 p-3">
        <Button variant="ghost" size="sm" className="w-full">
          View All Alerts
          <ChevronRight className="h-4 w-4" />
        </Button>
      </div>
    </Card>
  )
}
