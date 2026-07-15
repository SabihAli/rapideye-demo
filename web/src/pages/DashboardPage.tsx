import { Link } from 'react-router-dom'
import { VideoOff } from 'lucide-react'
import { CameraFeedCard } from '@/components/dashboard/CameraFeedCard'
import { AlertFeed } from '@/components/dashboard/AlertFeed'
import { Button } from '@/components/ui/Button'
import { toCameraDefinition } from '@/lib/cameras'
import { useCameras } from '@/context/CameraContext'
import { useDemo } from '@/context/DemoContext'

const API_PORT = import.meta.env.VITE_API_PORT ?? '8001'

export function DashboardPage() {
  const { cameras } = useCameras()
  const { health, healthError, alerts } = useDemo()
  const onlineCount = health?.streams.filter((s) => s.is_active).length ?? 0
  // A single camera in a 2-col tile grid leaves an awkward empty cell next
  // to it — drop to 1 column in that case. 2+ cameras keep wrapping at
  // md:grid-cols-2 same as before (this was never a 4-column tile grid —
  // the outer xl:grid-cols-4 below is a fixed 75/25 split against the alert
  // sidebar, unrelated to camera count).
  const tileGridColsClass = cameras.length === 1 ? 'md:grid-cols-1' : 'md:grid-cols-2'

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-bold text-text">Security Monitor</h1>
        <p className="text-sm text-muted">
          {cameras.length === 0
            ? 'No camera feeds added yet'
            : `Live annotated feeds from ${cameras.length} camera stream${cameras.length === 1 ? '' : 's'}`}
        </p>
      </div>

      {healthError && (
        <div className="rounded-xl border border-danger/30 bg-danger/10 px-4 py-3 text-sm text-danger">
          Backend unreachable: {healthError}. Start the API server with{' '}
          <code className="rounded bg-black/20 px-1">make backend</code> (port {API_PORT}).
        </div>
      )}

      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-text">Live Camera Feeds</h2>
        {cameras.length > 0 && (
          <span className="text-xs text-muted">
            {onlineCount}/{cameras.length} streams active
          </span>
        )}
      </div>

      {cameras.length === 0 ? (
        <div className="flex flex-col items-center gap-3 rounded-2xl border border-white/5 bg-bg-card px-4 py-12 text-center">
          <VideoOff className="h-8 w-8 text-muted" />
          <p className="text-sm text-muted">
            Add a camera to start viewing live feeds.
          </p>
          <Link to="/manage-cameras">
            <Button>Manage Cameras</Button>
          </Link>
        </div>
      ) : (
        <div className="grid grid-cols-1 items-start gap-6 xl:grid-cols-4">
          <div className={`grid grid-cols-1 gap-4 xl:col-span-3 ${tileGridColsClass}`}>
            {cameras.map((camera) => (
              <CameraFeedCard key={camera.camera_id} camera={toCameraDefinition(camera)} />
            ))}
          </div>

          <div className="xl:sticky xl:top-6 xl:h-[calc(100vh-3rem)]">
            <AlertFeed alerts={alerts} />
          </div>
        </div>
      )}

      {health && (
        <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
          {[
            {
              label: 'GPU',
              value: health.gpu_device ?? (health.gpu_available ? 'CUDA active' : 'CPU only'),
            },
            {
              label: 'Models Loaded',
              value: `${health.models_loaded.length} models`,
            },
            {
              label: 'Avg Stream FPS',
              value:
                health.streams.length > 0
                  ? (
                      health.streams.reduce((sum, s) => sum + s.current_fps, 0) / health.streams.length
                    ).toFixed(1)
                  : '—',
            },
            {
              label: 'Decode Errors',
              value: health.streams.reduce((sum, s) => sum + s.error_count, 0),
            },
          ].map((item) => (
            <div
              key={item.label}
              className="rounded-2xl border border-white/5 bg-bg-card px-4 py-3"
            >
              <p className="text-xs text-muted">{item.label}</p>
              <p className="mt-1 text-sm font-semibold text-text">{item.value}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
