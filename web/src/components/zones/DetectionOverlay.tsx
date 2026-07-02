import type { Detection } from '@/data/mockData'

interface DetectionOverlayProps {
  detections: Detection[]
}

export function DetectionOverlay({ detections }: DetectionOverlayProps) {
  return (
    <>
      {detections.map((det) => (
        <div
          key={det.id}
          className="pointer-events-none absolute border-2"
          style={{
            left: `${det.x}%`,
            top: `${det.y}%`,
            width: `${det.width}%`,
            height: `${det.height}%`,
            borderColor: det.color,
          }}
        >
          <span
            className="absolute -top-5 left-0 whitespace-nowrap rounded px-1 py-0.5 text-[10px] font-medium text-white"
            style={{ backgroundColor: det.color }}
          >
            {det.label} {det.confidence}%
          </span>
        </div>
      ))}
    </>
  )
}
