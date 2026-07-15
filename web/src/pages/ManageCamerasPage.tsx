import { useState } from 'react'
import { Plus, Video, VideoOff } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import { ConfirmDialog } from '@/components/ui/ConfirmDialog'
import { AddCameraModal } from '@/components/camera/AddCameraModal'
import { useCameras, MAX_CAMERAS } from '@/context/CameraContext'

export function ManageCamerasPage() {
  const { cameras, loading, error, removeCamera } = useCameras()
  const [showAdd, setShowAdd] = useState(false)
  const [removingId, setRemovingId] = useState<number | null>(null)
  const [removeError, setRemoveError] = useState<string | null>(null)
  const [removing, setRemoving] = useState(false)

  const atLimit = cameras.length >= MAX_CAMERAS
  const removingCamera = cameras.find((c) => c.camera_id === removingId) ?? null

  const handleConfirmRemove = async () => {
    if (removingId === null) return
    setRemoving(true)
    try {
      await removeCamera(removingId)
      setRemovingId(null)
      setRemoveError(null)
    } catch (err) {
      setRemoveError(err instanceof Error ? err.message : 'Failed to remove camera')
    } finally {
      setRemoving(false)
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-bold text-text">Manage Cameras</h1>
          <p className="text-sm text-muted">
            Add or remove camera feeds — up to {MAX_CAMERAS} at a time
          </p>
        </div>
        <Button
          onClick={() => setShowAdd(true)}
          disabled={atLimit}
          title={atLimit ? `You've reached the ${MAX_CAMERAS}-camera limit. Remove one to add another.` : undefined}
        >
          <Plus className="h-4 w-4" />
          Add Camera
        </Button>
      </div>

      {(error || removeError) && (
        <div className="rounded-xl border border-danger/30 bg-danger/10 px-4 py-3 text-sm text-danger">
          {error ?? removeError}
        </div>
      )}

      <Card>
        <CardHeader className="border-b border-white/5 pb-4">
          <CardTitle className="text-base">Registered Cameras</CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          {loading && cameras.length === 0 ? (
            <div className="px-4 py-10 text-center text-sm text-muted">Loading cameras…</div>
          ) : cameras.length === 0 ? (
            <div className="flex flex-col items-center gap-3 px-4 py-12 text-center">
              <VideoOff className="h-8 w-8 text-muted" />
              <p className="text-sm text-muted">
                No cameras added yet. Add a stream URL or upload a video file to get started.
              </p>
              <Button onClick={() => setShowAdd(true)}>
                <Plus className="h-4 w-4" />
                Add Your First Camera
              </Button>
            </div>
          ) : (
            <div className="divide-y divide-white/5">
              {cameras.map((camera) => (
                <div
                  key={camera.camera_id}
                  className="flex flex-wrap items-center justify-between gap-3 px-4 py-3"
                >
                  <div className="flex min-w-0 items-center gap-3">
                    <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-bg-secondary">
                      <Video className="h-4 w-4 text-muted" />
                    </div>
                    <div className="min-w-0 space-y-1">
                      <div className="flex flex-wrap items-center gap-2">
                        <p className="truncate text-sm font-medium text-text">{camera.name}</p>
                        <Badge variant="outline">{camera.source_type === 'url' ? 'URL' : 'File'}</Badge>
                        <Badge variant={camera.is_active ? 'success' : 'warning'}>
                          {camera.is_active ? 'Online' : 'Offline'}
                        </Badge>
                      </div>
                      <p className="truncate text-xs text-muted">
                        Camera {camera.camera_id}
                        {camera.is_active && <> · {camera.current_fps.toFixed(1)} FPS</>}
                        {camera.source_type === 'file' && camera.original_filename && (
                          <> · {camera.original_filename}</>
                        )}
                      </p>
                    </div>
                  </div>

                  <Button
                    variant="danger"
                    size="sm"
                    onClick={() => {
                      setRemoveError(null)
                      setRemovingId(camera.camera_id)
                    }}
                  >
                    Remove
                  </Button>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      <AddCameraModal open={showAdd} onClose={() => setShowAdd(false)} />

      <ConfirmDialog
        open={removingId !== null}
        title={`Remove ${removingCamera?.name ?? 'camera'}?`}
        description="This stops the live stream and permanently deletes this camera's zone configuration and detection toggles. Past recordings and alert history are kept."
        confirmLabel="Remove"
        confirming={removing}
        onConfirm={() => void handleConfirmRemove()}
        onCancel={() => setRemovingId(null)}
      />
    </div>
  )
}
