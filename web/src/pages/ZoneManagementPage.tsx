import { useState, useRef, useCallback } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { Pentagon, Save, X, ArrowLeft } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import { CameraPreview } from '@/components/camera/CameraPreview'
import { ZoneOverlay } from '@/components/zones/ZoneOverlay'
import { PolygonDraftOverlay } from '@/components/zones/PolygonDraftOverlay'
import { DraftPointMarkers } from '@/components/zones/DraftPointMarkers'
import { ZonePointEditor } from '@/components/zones/ZonePointEditor'
import { DetectionOverlay } from '@/components/zones/DetectionOverlay'
import { ZoneList } from '@/components/zones/ZoneList'
import { cameraFeeds } from '@/data/mockData'
import { useZones } from '@/context/ZoneContext'
import { getZoneStates, toPercent } from '@/lib/zoneUtils'
import {
  ZONE_TYPES,
  ZONE_TYPE_COLORS,
  MAX_ZONE_POINTS,
  MIN_ZONE_POINTS,
  type Point,
  type Zone,
  type ZoneType,
} from '@/types/zone'

export function ZoneManagementPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const { getZonesByCamera, addZone, updateZone } = useZones()

  const cameraFromUrl = searchParams.get('camera')
  const selectedCameraId =
    cameraFeeds.find((c) => c.id === cameraFromUrl)?.id ?? cameraFeeds[0].id

  const [isDrawing, setIsDrawing] = useState(false)
  const [editingZoneId, setEditingZoneId] = useState<string | null>(null)
  const [placingPoints, setPlacingPoints] = useState(true)
  const [draftPoints, setDraftPoints] = useState<Point[]>([])
  const [cursorPoint, setCursorPoint] = useState<Point | null>(null)
  const [selectedPointIndex, setSelectedPointIndex] = useState<number | null>(null)
  const [zoneName, setZoneName] = useState('')
  const [zoneType, setZoneType] = useState<ZoneType>('Restricted Zone')
  const canvasRef = useRef<HTMLDivElement>(null)
  const lastClickTime = useRef(0)

  const camera = cameraFeeds.find((c) => c.id === selectedCameraId) ?? cameraFeeds[0]
  const cameraZones = getZonesByCamera(selectedCameraId)
  const displayZones = editingZoneId
    ? cameraZones.filter((z) => z.id !== editingZoneId)
    : cameraZones
  const { occupied, violated } = getZoneStates(displayZones, camera.detections)

  const isFormOpen = isDrawing || editingZoneId !== null
  const canSave = draftPoints.length >= MIN_ZONE_POINTS && zoneName.trim().length > 0

  const resetForm = useCallback(() => {
    setIsDrawing(false)
    setEditingZoneId(null)
    setPlacingPoints(true)
    setDraftPoints([])
    setCursorPoint(null)
    setSelectedPointIndex(null)
    setZoneName('')
    setZoneType('Restricted Zone')
  }, [])

  const startDrawing = () => {
    setEditingZoneId(null)
    setDraftPoints([])
    setCursorPoint(null)
    setSelectedPointIndex(null)
    setZoneName('')
    setZoneType('Restricted Zone')
    setPlacingPoints(true)
    setIsDrawing(true)
  }

  const startEditing = (zone: Zone) => {
    setIsDrawing(false)
    setEditingZoneId(zone.id)
    setDraftPoints([...zone.points])
    setZoneName(zone.name)
    setZoneType(zone.type)
    setPlacingPoints(false)
    setCursorPoint(null)
    setSelectedPointIndex(null)
  }

  const handleCanvasClick = (e: React.MouseEvent) => {
    if (!isFormOpen || !placingPoints || !canvasRef.current) return

    const now = Date.now()
    if (now - lastClickTime.current < 250) return
    lastClickTime.current = now
    e.stopPropagation()

    if (draftPoints.length >= MAX_ZONE_POINTS) return

    const rect = canvasRef.current.getBoundingClientRect()
    const point = toPercent(e.clientX, e.clientY, rect)

    const last = draftPoints[draftPoints.length - 1]
    if (last) {
      const dx = point.x - last.x
      const dy = point.y - last.y
      if (Math.sqrt(dx * dx + dy * dy) < 1.5) return
    }

    setSelectedPointIndex(null)
    setDraftPoints((prev) => [...prev, point])
  }

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isFormOpen || !placingPoints || draftPoints.length === 0 || !canvasRef.current) {
      setCursorPoint(null)
      return
    }
    const rect = canvasRef.current.getBoundingClientRect()
    setCursorPoint(toPercent(e.clientX, e.clientY, rect))
  }

  const handleUndo = () => {
    setDraftPoints((prev) => prev.slice(0, -1))
    setSelectedPointIndex(null)
  }

  const handleSave = () => {
    if (!canSave) return

    if (editingZoneId) {
      updateZone(editingZoneId, {
        name: zoneName.trim(),
        type: zoneType,
        points: draftPoints,
        color: ZONE_TYPE_COLORS[zoneType],
      })
    } else {
      addZone({
        cameraId: selectedCameraId,
        name: zoneName.trim(),
        type: zoneType,
        points: draftPoints,
        color: ZONE_TYPE_COLORS[zoneType],
        enabled: true,
      })
    }
    resetForm()
  }

  const handleCameraChange = (id: string) => {
    if (id === selectedCameraId) return
    if (isFormOpen && !window.confirm('Switch camera? Unsaved zone changes will be lost.')) {
      return
    }
    resetForm()
    setSearchParams({ camera: id })
  }

  const handlePointsChange = (points: Point[]) => {
    setDraftPoints(points)
    setSelectedPointIndex(null)
  }

  const handleRemoveSelectedPoint = () => {
    if (selectedPointIndex === null) return
    setDraftPoints(draftPoints.filter((_, i) => i !== selectedPointIndex))
    setSelectedPointIndex(null)
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Link to="/">
            <Button variant="ghost" size="sm">
              <ArrowLeft className="h-4 w-4" />
            </Button>
          </Link>
          <h1 className="text-xl font-bold text-text">Zone Management</h1>
        </div>
        <Badge variant="info">{cameraZones.length} zones on {camera.name}</Badge>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <select
          value={selectedCameraId}
          onChange={(e) => handleCameraChange(e.target.value)}
          className="rounded-full border border-white/10 bg-bg-secondary px-4 py-2 text-sm text-text focus:border-white/25 focus:outline-none"
        >
          {cameraFeeds.map((cam) => (
            <option key={cam.id} value={cam.id}>{cam.name}</option>
          ))}
        </select>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        {/* Camera — left */}
        <div className="lg:col-span-2">
          <div className="mb-2 flex items-center justify-between">
            <p className="text-sm font-semibold text-text">{camera.name}</p>
            <span className="text-xs text-muted">{camera.status} · {camera.fps} FPS</span>
          </div>
          <Card className="overflow-hidden">
            <div ref={canvasRef}>
              <CameraPreview
                camera={camera}
                cursor={isFormOpen && placingPoints ? 'crosshair' : 'default'}
                onClick={handleCanvasClick}
                onMouseMove={handleMouseMove}
              >
                <ZoneOverlay zones={displayZones} occupiedIds={occupied} violatedIds={violated} />
                {isFormOpen && draftPoints.length > 0 && (
                  <>
                    <PolygonDraftOverlay
                      points={draftPoints}
                      closed={draftPoints.length >= MIN_ZONE_POINTS}
                      cursorPoint={cursorPoint}
                    />
                    <DraftPointMarkers
                      points={draftPoints}
                      selectedIndex={selectedPointIndex}
                      onSelect={placingPoints ? setSelectedPointIndex : undefined}
                      interactive={placingPoints}
                    />
                  </>
                )}
                <DetectionOverlay detections={camera.detections} />
              </CameraPreview>
            </div>
          </Card>

          {isFormOpen && selectedPointIndex !== null && placingPoints && (
            <div className="mt-2 flex items-center gap-2 rounded-xl border border-white/10 bg-bg-secondary px-3 py-2 text-sm">
              <span className="text-text-secondary">Point {selectedPointIndex + 1} selected</span>
              <Button variant="outline" size="sm" onClick={() => setSelectedPointIndex(null)}>Deselect</Button>
              <Button variant="danger" size="sm" onClick={handleRemoveSelectedPoint}>Remove</Button>
            </div>
          )}
        </div>

        {/* Sidebar — right, always visible */}
        <div className="space-y-4 lg:sticky lg:top-4 lg:self-start">
          {/* Zone form */}
          <Card>
            <CardHeader className="border-b border-white/5 pb-3">
              <CardTitle className="text-base">
                {editingZoneId ? 'Edit Zone' : isDrawing ? 'New Zone' : 'Create Zone'}
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 p-4">
              {!isFormOpen ? (
                <Button onClick={startDrawing} className="w-full">
                  <Pentagon className="h-4 w-4" />
                  Draw Zone
                </Button>
              ) : (
                <>
                  <div>
                    <label className="mb-1 block text-xs font-medium text-text">Zone Name *</label>
                    <input
                      type="text"
                      placeholder="e.g. Server Room"
                      value={zoneName}
                      onChange={(e) => setZoneName(e.target.value)}
                      className="w-full rounded-xl border border-white/10 bg-bg-secondary px-3 py-2 text-sm text-text placeholder:text-muted focus:border-white/25 focus:outline-none"
                    />
                  </div>

                  <div>
                    <label className="mb-1 block text-xs font-medium text-text">Zone Type</label>
                    <select
                      value={zoneType}
                      onChange={(e) => setZoneType(e.target.value as ZoneType)}
                      className="w-full rounded-xl border border-white/10 bg-bg-secondary px-3 py-2 text-sm text-text focus:border-white/25 focus:outline-none"
                    >
                      {ZONE_TYPES.map((t) => (
                        <option key={t} value={t}>{t}</option>
                      ))}
                    </select>
                  </div>

                  <div className="flex gap-2">
                    {placingPoints ? (
                      <Button
                        variant="outline"
                        size="sm"
                        className="flex-1"
                        disabled={draftPoints.length < MIN_ZONE_POINTS}
                        onClick={() => { setPlacingPoints(false); setCursorPoint(null) }}
                      >
                        Done Points
                      </Button>
                    ) : (
                      <Button variant="outline" size="sm" className="flex-1" onClick={() => setPlacingPoints(true)}>
                        Add Points
                      </Button>
                    )}
                    <Button variant="outline" size="sm" onClick={resetForm}>
                      <X className="h-4 w-4" />
                    </Button>
                  </div>

                  {placingPoints && (
                    <ZonePointEditor
                      points={draftPoints}
                      selectedIndex={selectedPointIndex}
                      onSelectPoint={setSelectedPointIndex}
                      onChange={handlePointsChange}
                      onUndo={handleUndo}
                      onClear={resetForm}
                    />
                  )}

                  {!placingPoints && draftPoints.length > 0 && (
                    <p className="text-xs text-muted">{draftPoints.length} points placed</p>
                  )}

                  <Button onClick={handleSave} disabled={!canSave} className="w-full">
                    <Save className="h-4 w-4" />
                    {editingZoneId ? 'Update Zone' : 'Save Zone'}
                  </Button>

                  {!canSave && (
                    <p className="text-center text-xs text-muted">
                      {draftPoints.length < MIN_ZONE_POINTS
                        ? `Place ${MIN_ZONE_POINTS - draftPoints.length} more point(s) on camera`
                        : 'Enter a zone name'}
                    </p>
                  )}
                </>
              )}
            </CardContent>
          </Card>

          {/* Saved zones */}
          <Card>
            <CardHeader className="border-b border-white/5 pb-3">
              <CardTitle className="text-base">Saved Zones</CardTitle>
            </CardHeader>
            <CardContent className="max-h-[400px] overflow-y-auto p-3">
              <ZoneList
                zones={cameraZones}
                cameraName={camera.name}
                editingZoneId={editingZoneId}
                onEdit={startEditing}
              />
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
