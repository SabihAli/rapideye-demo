import {
  Search,
  Bell,
  User,
  Activity,
  Wifi,
} from 'lucide-react'
import { Badge } from '@/components/ui/Badge'
import { systemStatus } from '@/data/mockData'

export function Topbar() {
  return (
    <header className="flex h-16 items-center justify-between border-b border-border bg-bg-card px-6">
      <div className="relative max-w-md flex-1">
        <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted" />
        <input
          type="search"
          placeholder="Search cameras, events, employees..."
          className="w-full rounded-lg border border-border bg-bg-secondary py-2 pl-10 pr-4 text-sm text-text placeholder:text-muted focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
        />
      </div>

      <div className="flex items-center gap-4">
        <div className="hidden items-center gap-2 rounded-lg border border-success/30 bg-success/10 px-3 py-1.5 md:flex">
          <Activity className="h-3.5 w-3.5 text-success" />
          <span className="text-xs font-medium text-success">System Online</span>
          <span className="text-xs text-muted">|</span>
          <Wifi className="h-3.5 w-3.5 text-primary" />
          <span className="text-xs text-muted">{systemStatus.streamLatency}ms latency</span>
        </div>

        <button className="relative rounded-lg p-2 text-muted transition-colors hover:bg-bg-secondary hover:text-text">
          <Bell className="h-5 w-5" />
          <span className="absolute right-1.5 top-1.5 flex h-4 w-4 items-center justify-center rounded-full bg-danger text-[10px] font-bold text-white">
            7
          </span>
        </button>

        <div className="flex items-center gap-3 border-l border-border pl-4">
          <div className="hidden text-right sm:block">
            <p className="text-sm font-medium text-text">Admin User</p>
            <p className="text-xs text-muted">Super Admin</p>
          </div>
          <div className="flex h-9 w-9 items-center justify-center rounded-full bg-primary/15">
            <User className="h-4 w-4 text-primary" />
          </div>
          <Badge variant="info" className="hidden lg:inline-flex">RBAC</Badge>
        </div>
      </div>
    </header>
  )
}
