import { useEffect } from 'react'
import { X } from 'lucide-react'

interface VideoModalProps {
  src: string | null
  onClose: () => void
}

export function VideoModal({ src, onClose }: VideoModalProps) {
  useEffect(() => {
    if (!src) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [src, onClose])

  if (!src) return null

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4"
      onClick={onClose}
    >
      <div
        className="relative w-full max-w-3xl overflow-hidden rounded-xl bg-bg-card shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <button
          onClick={onClose}
          className="absolute right-3 top-3 z-10 rounded-full bg-black/60 p-1.5 text-white transition-colors hover:bg-black/80"
          aria-label="Close"
        >
          <X className="h-4 w-4" />
        </button>
        {/* key forces a fresh <video> element per src so switching clips doesn't keep playing the old one */}
        <video key={src} src={src} controls autoPlay className="max-h-[80vh] w-full bg-black" />
      </div>
    </div>
  )
}
