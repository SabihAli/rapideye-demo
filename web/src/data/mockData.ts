import type { Zone } from '@/types/zone'

export type Severity = 'Critical' | 'High' | 'Medium' | 'Low'
export type AlertStatus = 'New' | 'Reviewing' | 'Resolved'
export type CameraStatus = 'Online' | 'Offline' | 'Warning'

export interface DashboardStat {
  id: string
  label: string
  value: string | number
  change?: string
  trend?: 'up' | 'down' | 'neutral'
  icon: string
  color: string
}

export interface Detection {
  id: string
  label: string
  confidence: number
  x: number
  y: number
  width: number
  height: number
  color: string
}

export interface ZoneOverlay {
  id: string
  name: string
  points: string
  color: string
  type: string
}

export interface CameraFeed {
  id: string
  name: string
  location: string
  status: CameraStatus
  aiProcessing: boolean
  fps: number
  detections: Detection[]
  gradient: string
  timestamp: string
}

export interface Alert {
  id: string
  title: string
  description: string
  severity: Severity
  camera: string
  zone: string
  timestamp: string
  status: AlertStatus
  eventType: string
  icon: string
}

export const dashboardStats: DashboardStat[] = [
  {
    id: 'cameras-online',
    label: 'Cameras Online',
    value: 18,
    change: '+2 today',
    trend: 'up',
    icon: 'camera',
    color: 'text-primary',
  },
  {
    id: 'active-alerts',
    label: 'Active Alerts',
    value: 7,
    change: '3 critical',
    trend: 'up',
    icon: 'alert',
    color: 'text-danger',
  },
  {
    id: 'persons-detected',
    label: 'Persons Detected Today',
    value: 342,
    change: '+12% vs yesterday',
    trend: 'up',
    icon: 'users',
    color: 'text-accent',
  },
  {
    id: 'restricted-events',
    label: 'Restricted Zone Events',
    value: 4,
    change: '2 unresolved',
    trend: 'neutral',
    icon: 'shield',
    color: 'text-warning',
  },
  {
    id: 'attendance',
    label: 'Attendance Marked',
    value: 89,
    change: 'of 95 employees',
    trend: 'neutral',
    icon: 'check',
    color: 'text-success',
  },
  {
    id: 'system-health',
    label: 'System Health',
    value: '98%',
    change: 'All services OK',
    trend: 'up',
    icon: 'activity',
    color: 'text-success',
  },
]

export const cameraFeeds: CameraFeed[] = [
  {
    id: 'cam-1',
    name: 'Camera 1',
    location: '',
    status: 'Online',
    aiProcessing: true,
    fps: 30,
    gradient: 'from-slate-800 via-slate-700 to-slate-900',
    timestamp: '14:32:08',
    detections: [
      { id: 'd1', label: 'Person', confidence: 92, x: 15, y: 25, width: 18, height: 45, color: '#38BDF8' },
      { id: 'd2', label: 'Person', confidence: 88, x: 55, y: 30, width: 16, height: 42, color: '#38BDF8' },
      { id: 'd3', label: 'Vehicle', confidence: 76, x: 72, y: 50, width: 25, height: 30, color: '#14B8A6' },
    ],
  },
  {
    id: 'cam-2',
    name: 'Camera 2',
    location: '',
    status: 'Online',
    aiProcessing: true,
    fps: 25,
    gradient: 'from-gray-800 via-gray-700 to-gray-900',
    timestamp: '14:32:06',
    detections: [
      { id: 'd4', label: 'Sarah Khan', confidence: 97, x: 30, y: 20, width: 14, height: 40, color: '#22C55E' },
      { id: 'd5', label: 'Unknown', confidence: 84, x: 65, y: 35, width: 15, height: 38, color: '#F59E0B' },
    ],
  },
  {
    id: 'cam-3',
    name: 'Camera 3',
    location: '',
    status: 'Warning',
    aiProcessing: true,
    fps: 22,
    gradient: 'from-stone-800 via-stone-700 to-stone-900',
    timestamp: '14:32:04',
    detections: [
      { id: 'd6', label: 'Fire', confidence: 87, x: 40, y: 15, width: 20, height: 25, color: '#EF4444' },
      { id: 'd7', label: 'No Helmet', confidence: 81, x: 20, y: 25, width: 14, height: 35, color: '#F59E0B' },
      { id: 'd8', label: 'Person', confidence: 90, x: 70, y: 45, width: 16, height: 38, color: '#38BDF8' },
    ],
  },
  {
    id: 'cam-4',
    name: 'Camera 4',
    location: '',
    status: 'Online',
    aiProcessing: true,
    fps: 28,
    gradient: 'from-zinc-800 via-zinc-700 to-zinc-900',
    timestamp: '14:32:10',
    detections: [
      { id: 'd9', label: 'Person', confidence: 94, x: 10, y: 30, width: 12, height: 35, color: '#38BDF8' },
      { id: 'd10', label: 'Person', confidence: 91, x: 25, y: 28, width: 13, height: 36, color: '#38BDF8' },
      { id: 'd11', label: 'Person', confidence: 89, x: 40, y: 32, width: 12, height: 34, color: '#38BDF8' },
      { id: 'd12', label: 'Crowd', confidence: 85, x: 8, y: 25, width: 50, height: 45, color: '#F59E0B' },
    ],
  },
]

export const initialZones: Zone[] = [
  {
    id: 'z1',
    cameraId: 'cam-1',
    name: 'Entry Gate',
    type: 'Entry Gate',
    points: [{ x: 5, y: 60 }, { x: 45, y: 60 }, { x: 45, y: 95 }, { x: 5, y: 95 }],
    color: '#14B8A6',
    enabled: true,
    createdAt: '2026-01-01T00:00:00Z',
  },
  {
    id: 'z2',
    cameraId: 'cam-1',
    name: 'Restricted',
    type: 'Restricted Zone',
    points: [{ x: 60, y: 10 }, { x: 95, y: 10 }, { x: 95, y: 50 }, { x: 60, y: 50 }],
    color: '#EF4444',
    enabled: true,
    createdAt: '2026-01-01T00:00:00Z',
  },
  {
    id: 'z3',
    cameraId: 'cam-2',
    name: 'Reception Desk',
    type: 'Production Area',
    points: [{ x: 10, y: 70 }, { x: 90, y: 70 }, { x: 90, y: 95 }, { x: 10, y: 95 }],
    color: '#38BDF8',
    enabled: true,
    createdAt: '2026-01-01T00:00:00Z',
  },
  {
    id: 'z4',
    cameraId: 'cam-3',
    name: 'PPE Zone',
    type: 'PPE Required Zone',
    points: [{ x: 5, y: 5 }, { x: 95, y: 5 }, { x: 95, y: 55 }, { x: 5, y: 55 }],
    color: '#F59E0B',
    enabled: true,
    createdAt: '2026-01-01T00:00:00Z',
  },
  {
    id: 'z5',
    cameraId: 'cam-3',
    name: 'Fire Risk',
    type: 'Fire Risk Zone',
    points: [{ x: 35, y: 10 }, { x: 65, y: 10 }, { x: 65, y: 40 }, { x: 35, y: 40 }],
    color: '#EF4444',
    enabled: true,
    createdAt: '2026-01-01T00:00:00Z',
  },
  {
    id: 'z6',
    cameraId: 'cam-4',
    name: 'Parking Entry',
    type: 'Entry Gate',
    points: [{ x: 0, y: 40 }, { x: 30, y: 40 }, { x: 30, y: 100 }, { x: 0, y: 100 }],
    color: '#14B8A6',
    enabled: true,
    createdAt: '2026-01-01T00:00:00Z',
  },
]

export const recentAlerts: Alert[] = [
  {
    id: 'alert-1',
    title: 'Fire detected in Warehouse',
    description: 'AI model detected fire signature with 87% confidence',
    severity: 'Critical',
    camera: 'Camera 3',
    zone: 'Fire Risk Zone',
    timestamp: '2 min ago',
    status: 'New',
    eventType: 'Fire Detection',
    icon: 'flame',
  },
  {
    id: 'alert-2',
    title: 'Unknown person after office hours',
    description: 'Unregistered face detected at 9:14 PM',
    severity: 'High',
    camera: 'Camera 2',
    zone: 'Reception Desk',
    timestamp: '8 min ago',
    status: 'Reviewing',
    eventType: 'Unknown Person',
    icon: 'user-x',
  },
  {
    id: 'alert-3',
    title: 'Person entered restricted zone',
    description: 'Unauthorized access to server room area',
    severity: 'Critical',
    camera: 'Camera 1',
    zone: 'Restricted Zone',
    timestamp: '15 min ago',
    status: 'New',
    eventType: 'Zone Violation',
    icon: 'shield-alert',
  },
  {
    id: 'alert-4',
    title: 'Helmet missing in PPE zone',
    description: 'Worker detected without required safety helmet',
    severity: 'High',
    camera: 'Camera 3',
    zone: 'PPE Zone',
    timestamp: '22 min ago',
    status: 'Reviewing',
    eventType: 'PPE Violation',
    icon: 'hard-hat',
  },
  {
    id: 'alert-5',
    title: 'Crowd detected near main entrance',
    description: '8+ persons gathered — possible congestion',
    severity: 'Medium',
    camera: 'Camera 4',
    zone: 'Parking Entry',
    timestamp: '35 min ago',
    status: 'Resolved',
    eventType: 'Crowd Detection',
    icon: 'users',
  },
  {
    id: 'alert-6',
    title: 'Camera 3 connection unstable',
    description: 'Stream latency exceeded 500ms threshold',
    severity: 'High',
    camera: 'Camera 3',
    zone: '—',
    timestamp: '1 hr ago',
    status: 'Reviewing',
    eventType: 'System Alert',
    icon: 'wifi-off',
  },
]

export const navItems = [
  { label: 'Dashboard', path: '/', icon: 'layout-dashboard' },
  { label: 'Live Cameras', path: '/live-cameras', icon: 'video' },
  { label: 'Camera Management', path: '/cameras', icon: 'camera' },
  { label: 'Zone Management', path: '/zones', icon: 'map' },
  { label: 'Alerts', path: '/alerts', icon: 'bell' },
  { label: 'Event Logs', path: '/events', icon: 'list' },
  { label: 'Attendance Reports', path: '/attendance', icon: 'clipboard-list' },
  { label: 'Employees / Users', path: '/employees', icon: 'users' },
  { label: 'Face Profiles', path: '/face-profiles', icon: 'scan-face' },
  { label: 'Alert Rules', path: '/alert-rules', icon: 'settings-2' },
  { label: 'Access Control', path: '/access-control', icon: 'lock' },
  { label: 'System Health', path: '/system-health', icon: 'heart-pulse' },
  { label: 'Roles & Permissions', path: '/roles', icon: 'shield' },
  { label: 'Settings', path: '/settings', icon: 'settings' },
]

export const systemStatus = {
  aiWorkers: 4,
  aiWorkersActive: 4,
  storageUsed: 67,
  cpuUsage: 42,
  gpuUsage: 58,
  streamLatency: 120,
  uptime: '99.8%',
}
