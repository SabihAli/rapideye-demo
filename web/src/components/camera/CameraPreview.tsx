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
}: CameraPreviewProps) {
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
        <img
          src={frameSrc}
          alt="Live camera feed"
          className="absolute inset-0 h-full w-full object-contain object-center"
        />
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
        </>
      )}
      {children}
    </div>
  )
}
