export type ZoneType =
  | 'Restricted Zone'
  | 'Entry Gate'
  | 'Exit Gate'
  | 'PPE Required Zone'
  | 'Fire Risk Zone'
  | 'No-Entry Area'
  | 'Production Area'
  | 'Lab Area'

export interface Point {
  x: number
  y: number
}

export interface Zone {
  id: string
  cameraId: string
  name: string
  type: ZoneType
  points: Point[]
  color: string
  enabled: boolean
  createdAt: string
}

export const MIN_ZONE_POINTS = 3
export const MAX_ZONE_POINTS = 12

export const ZONE_TYPES: ZoneType[] = [
  'Restricted Zone',
  'Entry Gate',
  'Exit Gate',
  'PPE Required Zone',
  'Fire Risk Zone',
  'No-Entry Area',
  'Production Area',
  'Lab Area',
]

export const ZONE_TYPE_COLORS: Record<ZoneType, string> = {
  'Restricted Zone': '#EF4444',
  'Entry Gate': '#14B8A6',
  'Exit Gate': '#38BDF8',
  'PPE Required Zone': '#F59E0B',
  'Fire Risk Zone': '#EF4444',
  'No-Entry Area': '#EF4444',
  'Production Area': '#38BDF8',
  'Lab Area': '#A78BFA',
}
