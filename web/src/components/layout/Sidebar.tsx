import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard,
  Video,
  Camera,
  Map,
  Bell,
  List,
  ClipboardList,
  Users,
  ScanFace,
  Settings2,
  Lock,
  HeartPulse,
  Shield,
  Settings,
  ChevronLeft,
  ChevronRight,
  Eye,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { navItems } from '@/data/mockData'
import { useState } from 'react'

const iconMap: Record<string, React.ElementType> = {
  'layout-dashboard': LayoutDashboard,
  video: Video,
  camera: Camera,
  map: Map,
  bell: Bell,
  list: List,
  'clipboard-list': ClipboardList,
  users: Users,
  'scan-face': ScanFace,
  'settings-2': Settings2,
  lock: Lock,
  'heart-pulse': HeartPulse,
  shield: Shield,
  settings: Settings,
}

export function Sidebar() {
  const [collapsed, setCollapsed] = useState(false)

  return (
    <aside
      className={cn(
        'flex h-screen flex-col border-r border-border bg-bg-card transition-all duration-300',
        collapsed ? 'w-16' : 'w-64'
      )}
    >
      <div className="flex h-16 items-center gap-3 border-b border-border px-4">
        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-primary/15">
          <Eye className="h-5 w-5 text-primary" />
        </div>
        {!collapsed && (
          <div className="min-w-0">
            <p className="truncate text-sm font-bold text-text">SecureVision AI</p>
            <p className="truncate text-xs text-muted">Command Center</p>
          </div>
        )}
      </div>

      <nav className="flex-1 overflow-y-auto py-3">
        <ul className="space-y-0.5 px-2">
          {navItems.map((item) => {
            const Icon = iconMap[item.icon] || LayoutDashboard
            return (
              <li key={item.path}>
                <NavLink
                  to={item.path}
                  end={item.path === '/'}
                  className={({ isActive }) =>
                    cn(
                      'flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm transition-colors',
                      isActive
                        ? 'bg-primary/10 text-primary font-medium'
                        : 'text-muted hover:bg-bg-secondary hover:text-text'
                    )
                  }
                  title={collapsed ? item.label : undefined}
                >
                  <Icon className="h-4 w-4 shrink-0" />
                  {!collapsed && <span className="truncate">{item.label}</span>}
                </NavLink>
              </li>
            )
          })}
        </ul>
      </nav>

      <div className="border-t border-border p-3">
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="flex w-full items-center justify-center rounded-lg p-2 text-muted transition-colors hover:bg-bg-secondary hover:text-text"
        >
          {collapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
        </button>
      </div>
    </aside>
  )
}
