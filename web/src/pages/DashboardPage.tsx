import { RefreshCw } from 'lucide-react'
import { CameraFeedCard } from '@/components/dashboard/CameraFeedCard'
import { AlertFeed } from '@/components/dashboard/AlertFeed'
import { Button } from '@/components/ui/Button'
import { CAMERAS } from '@/lib/cameras'
import { useDemo } from '@/context/DemoContext'

export function DashboardPage() {
  const { health, healthError, alerts, startDemo, startingDemo } = useDemo()
  const onlineCount = health?.streams.filter((s) => s.is_active).length ?? 0

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-bold text-text">Security Monitor</h1>
          <p className="text-sm text-muted">Live annotated feeds from 4 camera streams</p>
        </div>
        <Button onClick={startDemo} disabled={startingDemo}>
          <RefreshCw className={`h-4 w-4 ${startingDemo ? 'animate-spin' : ''}`} />
          {startingDemo ? 'Resetting…' : 'Start / Reset Demo'}
        </Button>
      </div>

      {healthError && (
        <div className="rounded-xl border border-danger/30 bg-danger/10 px-4 py-3 text-sm text-danger">
          Backend unreachable: {healthError}. Start the API server on port 8000.
        </div>
      )}

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-4">
        <div className="space-y-4 xl:col-span-3">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-text">Live Camera Feeds</h2>
            <span className="text-xs text-muted">
              {onlineCount}/{CAMERAS.length} streams active
            </span>
          </div>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            {CAMERAS.map((camera) => (
              <CameraFeedCard key={camera.id} camera={camera} />
            ))}
          </div>
        </div>

        <div className="xl:sticky xl:top-6 xl:h-[calc(100vh-3rem)]">
          <AlertFeed alerts={alerts} />
        </div>
      </div>

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
