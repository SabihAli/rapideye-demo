import { CameraFeedCard } from '@/components/dashboard/CameraFeedCard'
import { AlertFeed } from '@/components/dashboard/AlertFeed'
import { cameraFeeds, recentAlerts, systemStatus } from '@/data/mockData'

export function DashboardPage() {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-6 xl:grid-cols-4">
        <div className="space-y-4 xl:col-span-3">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-text">Live Camera Feeds</h2>
            <span className="text-xs text-muted">
              {cameraFeeds.filter((c) => c.status === 'Online').length}/{cameraFeeds.length} online
            </span>
          </div>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            {cameraFeeds.map((camera) => (
              <CameraFeedCard key={camera.id} camera={camera} />
            ))}
          </div>
        </div>

        <div className="xl:sticky xl:top-6 xl:h-[calc(100vh-3rem)]">
          <AlertFeed alerts={recentAlerts} />
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        {[
          { label: 'AI Workers', value: `${systemStatus.aiWorkersActive}/${systemStatus.aiWorkers} active` },
          { label: 'Storage Used', value: `${systemStatus.storageUsed}%` },
          { label: 'GPU Usage', value: `${systemStatus.gpuUsage}%` },
          { label: 'Uptime', value: systemStatus.uptime },
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
    </div>
  )
}
