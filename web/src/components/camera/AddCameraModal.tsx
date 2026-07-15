import { useEffect, useRef, useState } from 'react'
import { Link2, UploadCloud, X } from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { cn } from '@/lib/utils'
import { useCameras } from '@/context/CameraContext'

const ALLOWED_EXTENSIONS = ['.mp4', '.mov', '.avi', '.mkv']
const MAX_UPLOAD_BYTES = 500 * 1024 * 1024

type SourceType = 'url' | 'file'

interface AddCameraModalProps {
  open: boolean
  onClose: () => void
}

function formatBytes(bytes: number): string {
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function fileValidationError(file: File): string | null {
  const ext = file.name.slice(file.name.lastIndexOf('.')).toLowerCase()
  if (!ALLOWED_EXTENSIONS.includes(ext)) {
    return `Unsupported file type "${ext}". Use MP4, MOV, AVI, or MKV.`
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    return `File is ${formatBytes(file.size)} — the limit is ${formatBytes(MAX_UPLOAD_BYTES)}.`
  }
  return null
}

export function AddCameraModal({ open, onClose }: AddCameraModalProps) {
  const { addCameraByUrl, addCameraByFile } = useCameras()
  const [sourceType, setSourceType] = useState<SourceType>('url')
  const [name, setName] = useState('')
  const [url, setUrl] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [fileError, setFileError] = useState<string | null>(null)
  const [dragActive, setDragActive] = useState(false)
  const [submitError, setSubmitError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [progress, setProgress] = useState<number | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const reset = () => {
    setSourceType('url')
    setName('')
    setUrl('')
    setFile(null)
    setFileError(null)
    setDragActive(false)
    setSubmitError(null)
    setSubmitting(false)
    setProgress(null)
  }

  const handleClose = () => {
    if (submitting) return
    reset()
    onClose()
  }

  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') handleClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, submitting])

  if (!open) return null

  const handleFileSelected = (selected: File | null) => {
    if (!selected) {
      setFile(null)
      setFileError(null)
      return
    }
    setFile(selected)
    setFileError(fileValidationError(selected))
  }

  const canSubmit =
    name.trim().length > 0 &&
    !submitting &&
    (sourceType === 'url' ? url.trim().length > 0 : file !== null && fileError === null)

  const handleSubmit = async () => {
    if (!canSubmit) return
    setSubmitting(true)
    setSubmitError(null)
    try {
      if (sourceType === 'url') {
        await addCameraByUrl(name.trim(), url.trim())
      } else if (file) {
        setProgress(0)
        await addCameraByFile(name.trim(), file, setProgress)
      }
      reset()
      onClose()
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : 'Failed to add camera')
      setSubmitting(false)
      setProgress(null)
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4"
      onClick={handleClose}
    >
      <div
        className="w-full max-w-md rounded-2xl border border-white/5 bg-bg-card p-5 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between">
          <h2 className="text-base font-semibold text-text">Add Camera</h2>
          <button
            onClick={handleClose}
            className="rounded-full p-1 text-muted transition-colors hover:bg-white/5 hover:text-text"
            aria-label="Close"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="mt-4 space-y-4">
          <div>
            <label className="mb-1 block text-xs font-medium text-text">Camera Name *</label>
            <input
              type="text"
              placeholder="e.g. Front Lobby"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-full rounded-xl border border-white/10 bg-bg-secondary px-3 py-2 text-sm text-text placeholder:text-muted focus:border-white/25 focus:outline-none"
            />
          </div>

          <div className="grid grid-cols-2 gap-2">
            <button
              type="button"
              onClick={() => setSourceType('url')}
              className={cn(
                'flex items-center justify-center gap-1.5 rounded-full border px-3 py-1.5 text-xs font-medium transition-colors',
                sourceType === 'url'
                  ? 'border-white/25 bg-bg-secondary text-text'
                  : 'border-white/10 text-muted hover:text-text-secondary'
              )}
            >
              <Link2 className="h-3.5 w-3.5" />
              URL
            </button>
            <button
              type="button"
              onClick={() => setSourceType('file')}
              className={cn(
                'flex items-center justify-center gap-1.5 rounded-full border px-3 py-1.5 text-xs font-medium transition-colors',
                sourceType === 'file'
                  ? 'border-white/25 bg-bg-secondary text-text'
                  : 'border-white/10 text-muted hover:text-text-secondary'
              )}
            >
              <UploadCloud className="h-3.5 w-3.5" />
              Local File
            </button>
          </div>

          {sourceType === 'url' ? (
            <div>
              <label className="mb-1 block text-xs font-medium text-text">Stream URL *</label>
              <input
                type="text"
                placeholder="rtsp://192.168.1.50:554/stream1"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                className="w-full rounded-xl border border-white/10 bg-bg-secondary px-3 py-2 text-sm text-text placeholder:text-muted focus:border-white/25 focus:outline-none"
              />
            </div>
          ) : (
            <div>
              <label className="mb-1 block text-xs font-medium text-text">Video File *</label>
              <div
                onDragOver={(e) => {
                  e.preventDefault()
                  setDragActive(true)
                }}
                onDragLeave={() => setDragActive(false)}
                onDrop={(e) => {
                  e.preventDefault()
                  setDragActive(false)
                  handleFileSelected(e.dataTransfer.files[0] ?? null)
                }}
                className={cn(
                  'flex flex-col items-center gap-2 rounded-xl border border-dashed px-4 py-6 text-center transition-colors',
                  dragActive ? 'border-white/40 bg-bg-secondary' : 'border-white/15'
                )}
              >
                <UploadCloud className="h-6 w-6 text-muted" />
                {file ? (
                  <p className="text-xs text-text-secondary">
                    {file.name} · {formatBytes(file.size)}
                  </p>
                ) : (
                  <p className="text-xs text-muted">
                    Drag and drop a video file here, or
                  </p>
                )}
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => fileInputRef.current?.click()}
                >
                  Browse
                </Button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept={ALLOWED_EXTENSIONS.join(',')}
                  className="hidden"
                  onChange={(e) => handleFileSelected(e.target.files?.[0] ?? null)}
                />
              </div>
              <p className="mt-1 text-[11px] text-muted">
                MP4, MOV, AVI, or MKV — up to {formatBytes(MAX_UPLOAD_BYTES)}.
              </p>
              {fileError && <p className="mt-1 text-xs text-danger">{fileError}</p>}
            </div>
          )}

          {progress !== null && (
            <div>
              <div className="h-1.5 w-full overflow-hidden rounded-full bg-bg-secondary">
                <div
                  className="h-full rounded-full bg-primary transition-all"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <p className="mt-1 text-[11px] text-muted">Uploading… {progress}%</p>
            </div>
          )}

          {submitError && (
            <div className="rounded-lg border border-danger/30 bg-danger/10 px-3 py-2 text-xs text-danger">
              {submitError}
            </div>
          )}
        </div>

        <div className="mt-5 flex justify-end gap-2">
          <Button variant="outline" size="sm" onClick={handleClose} disabled={submitting}>
            Cancel
          </Button>
          <Button size="sm" onClick={() => void handleSubmit()} disabled={!canSubmit}>
            {submitting ? 'Adding…' : 'Add Camera'}
          </Button>
        </div>
      </div>
    </div>
  )
}
