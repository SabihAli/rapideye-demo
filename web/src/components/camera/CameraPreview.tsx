import { cn } from '@/lib/utils'
import type { CameraFeed } from '@/data/mockData'

interface CameraPreviewProps {
  camera: CameraFeed
  children?: React.ReactNode
  className?: string
  onMouseDown?: React.MouseEventHandler<HTMLDivElement>
  onClick?: React.MouseEventHandler<HTMLDivElement>
  onMouseMove?: React.MouseEventHandler<HTMLDivElement>
  onMouseUp?: React.MouseEventHandler<HTMLDivElement>
  cursor?: string
}

export function CameraPreview({
  camera,
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
      <div className={cn('absolute inset-0 bg-gradient-to-br', camera.gradient)} />
      <div
        className="absolute inset-0 opacity-20"
        style={{
          backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.5'/%3E%3C/svg%3E")`,
        }}
      />
      <div className="absolute bottom-0 left-0 right-0 h-1/3 bg-gradient-to-t from-black/60 to-transparent" />
      <div className="absolute left-[20%] top-[30%] h-[40%] w-[8%] rounded-sm bg-slate-600/40" />
      <div className="absolute left-[50%] top-[25%] h-[45%] w-[10%] rounded-sm bg-slate-500/40" />
      <div className="absolute right-[15%] top-[40%] h-[35%] w-[20%] rounded bg-slate-600/30" />
      {children}
    </div>
  )
}
