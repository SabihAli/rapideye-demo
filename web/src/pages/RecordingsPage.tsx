import { useEffect, useState } from 'react'
import { Circle, Play, RefreshCw } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import { getAllCameraRecordings, recordingUrl } from '@/lib/api'
import { CAMERAS, cameraFromNumId } from '@/lib/cameras'
import { formatTimestamp } from '@/context/DemoContext'
import type { CameraRecording } from '@/types/api'

function formatDuration(seconds?: number | null): string {
  if (seconds == null) return '—'
  if (seconds < 60) return `${seconds.toFixed(0)}s`
  const mins = Math.floor(seconds / 60)
  const secs = Math.round(seconds % 60)
  return `${mins}m ${secs}s`
}

function formatFileSize(bytes?: number | null): string {
  if (bytes == null) return '—'
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export function RecordingsPage() {
  const [recordings, setRecordings] = useState<CameraRecording[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const loadRecordings = async () => {
    setLoading(true)
    try {
      const data = await getAllCameraRecordings()
      setRecordings(data)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load recordings')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void loadRecordings()
  }, [])

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-bold text-text">Camera Recordings</h1>
          <p className="text-sm text-muted">
            Saved MP4 recordings from {CAMERAS.length} camera feeds
          </p>
        </div>
        <Button onClick={() => void loadRecordings()} disabled={loading} variant="outline">
          <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </Button>
      </div>

      {error && (
        <div className="rounded-xl border border-danger/30 bg-danger/10 px-4 py-3 text-sm text-danger">
          {error}
        </div>
      )}

      <Card>
        <CardHeader className="border-b border-white/5 pb-4">
          <CardTitle className="text-base">Recording Library</CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          {loading && recordings.length === 0 ? (
            <div className="px-4 py-10 text-center text-sm text-muted">Loading recordings…</div>
          ) : recordings.length === 0 ? (
            <div className="px-4 py-10 text-center text-sm text-muted">
              No recordings yet. Start recording from a live camera tile on the dashboard.
            </div>
          ) : (
            <div className="divide-y divide-white/5">
              {recordings.map((recording) => {
                const camera = cameraFromNumId(recording.camera_id)
                const isActive = recording.status === 'recording'

                return (
                  <div
                    key={recording.id}
                    className="flex flex-wrap items-center justify-between gap-3 px-4 py-3"
                  >
                    <div className="min-w-0 space-y-1">
                      <div className="flex flex-wrap items-center gap-2">
                        <p className="text-sm font-medium text-text">{camera.name}</p>
                        <Badge variant={isActive ? 'danger' : recording.status === 'completed' ? 'success' : 'warning'}>
                          {isActive ? (
                            <span className="inline-flex items-center gap-1">
                              <Circle className="h-2 w-2 fill-danger" />
                              Recording
                            </span>
                          ) : (
                            recording.status
                          )}
                        </Badge>
                      </div>
                      <p className="text-xs text-muted">
                        Started {formatTimestamp(recording.started_at)}
                        {recording.duration_seconds != null && (
                          <> · {formatDuration(recording.duration_seconds)}</>
                        )}
                        {recording.file_size_bytes != null && (
                          <> · {formatFileSize(recording.file_size_bytes)}</>
                        )}
                      </p>
                    </div>

                    {recording.status === 'completed' && (
                      <a
                        href={recordingUrl(recording.playback_url)}
                        target="_blank"
                        rel="noreferrer"
                        className="inline-flex items-center gap-1 rounded-lg border border-white/10 px-3 py-1.5 text-xs text-primary hover:border-white/20"
                      >
                        <Play className="h-3.5 w-3.5" />
                        Play
                      </a>
                    )}
                  </div>
                )
              })}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
