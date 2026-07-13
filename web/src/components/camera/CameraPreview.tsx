import { useState } from 'react'
import { cn } from '@/lib/utils'

interface CameraPreviewProps {
  frameSrc?: string | null
  gradient?: string
  children?: React.ReactNode
  className?: string
  onMouseDown?: React.MouseEventHandler<HTMLDivElement>
  onClick?: React.MouseEventHandler<HTMLDivElement>
  onMouseMove?: React.MouseEventHandler<HTMLDivElement>
  onMouseUp?: React.MouseEventHandler<HTMLDivElement>
  cursor?: string
  // Ref to the element whose bounding box exactly matches the visible video
  // content (see aspectRatio note below) — callers doing pixel-space math
  // (zone point placement) must measure against this, not the outer
  // aspect-video container, or coordinates land in letterboxed dead space.
  contentRef?: React.Ref<HTMLDivElement>
}

export function CameraPreview({
  frameSrc,
  gradient = 'from-slate-800 via-slate-700 to-slate-900',
  children,
  className,
  onMouseDown,
  onClick,
  onMouseMove,
  onMouseUp,
  cursor,
  contentRef,
}: CameraPreviewProps) {
  // The outer container is pinned to a 16:9 box, but source video isn't
  // always 16:9 (e.g. portrait clips) — object-contain then letterboxes the
  // <img> inside it. Overlay children (zone polygons, draft points) used to
  // be siblings sized to the full 16:9 box via inset-0/h-full/w-full, so on
  // any non-16:9 stream they were positioned against the letterboxed dead
  // space instead of the actual visible frame. Wrapping the image + overlays
  // together in a div sized to the stream's real aspect ratio (once known
  // from the first decoded frame) keeps both in the same coordinate space.
  const [aspectRatio, setAspectRatio] = useState<number | undefined>(undefined)

  return (
    <div
      className={cn(
        'relative aspect-video overflow-hidden bg-bg-secondary cctv-scanline select-none',
        className
      )}
      style={{ cursor }}
      onMouseDown={onMouseDown}
      onClick={onClick}
      onMouseMove={onMouseMove}
      onMouseUp={onMouseUp}
      onMouseLeave={onMouseUp}
    >
      {frameSrc ? (
        <div
          ref={contentRef}
          className="absolute inset-0 m-auto max-h-full max-w-full"
          style={{ aspectRatio }}
        >
          <img
            src={frameSrc}
            alt="Live camera feed"
            className="block h-full w-full"
            onLoad={(e) => {
              const { naturalWidth, naturalHeight } = e.currentTarget
              if (naturalWidth && naturalHeight) {
                setAspectRatio(naturalWidth / naturalHeight)
              }
            }}
          />
          {children}
        </div>
      ) : (
        <>
          <div className={cn('absolute inset-0 bg-gradient-to-br', gradient)} />
          <div
            className="absolute inset-0 opacity-20"
            style={{
              backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.5'/%3E%3C/svg%3E")`,
            }}
          />
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="rounded bg-black/50 px-3 py-1 text-xs text-muted">Waiting for stream…</span>
          </div>
          {children}
        </>
      )}
    </div>
  )
}
