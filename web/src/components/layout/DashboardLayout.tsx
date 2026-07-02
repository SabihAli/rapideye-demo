import { Outlet, Link, useLocation } from 'react-router-dom'
import { LayoutDashboard, Map } from 'lucide-react'
import { cn } from '@/lib/utils'

const navLinks = [
  { label: 'Dashboard', path: '/', icon: LayoutDashboard },
  { label: 'Zone Management', path: '/zones', icon: Map },
]

export function DashboardLayout() {
  const location = useLocation()

  return (
    <div className="min-h-screen overflow-hidden bg-bg-main">
      <nav className="flex items-center gap-2 border-b border-white/5 bg-bg-card px-6 py-3">
        {navLinks.map(({ label, path, icon: Icon }) => (
          <Link
            key={path}
            to={path}
            className={cn(
              'flex items-center gap-2 rounded-full px-4 py-2 text-sm transition-colors',
              location.pathname === path || (path !== '/' && location.pathname.startsWith(path))
                ? 'bg-bg-secondary text-text font-medium'
                : 'text-muted hover:bg-bg-secondary/60 hover:text-text-secondary'
            )}
          >
            <Icon className="h-4 w-4" />
            {label}
          </Link>
        ))}
      </nav>
      <main className="overflow-y-auto p-6 md:p-8">
        <Outlet />
      </main>
    </div>
  )
}
