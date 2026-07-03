import {
  Camera,
  AlertTriangle,
  Users,
  Shield,
  CheckCircle,
  Activity,
  TrendingUp,
  TrendingDown,
  Minus,
} from 'lucide-react'
import { Card, CardContent } from '@/components/ui/Card'
import { dashboardStats } from '@/data/mockData'
import { cn } from '@/lib/utils'

const iconMap: Record<string, React.ElementType> = {
  camera: Camera,
  alert: AlertTriangle,
  users: Users,
  shield: Shield,
  check: CheckCircle,
  activity: Activity,
}

export function StatsCards() {
  return (
    <div className="grid grid-cols-2 gap-4 md:grid-cols-3 xl:grid-cols-6">
      {dashboardStats.map((stat) => {
        const Icon = iconMap[stat.icon] || Activity
        const TrendIcon =
          stat.trend === 'up' ? TrendingUp : stat.trend === 'down' ? TrendingDown : Minus

        return (
          <Card
            key={stat.id}
            className="group transition-all duration-200 hover:border-primary/30 hover:shadow-primary/5"
          >
            <CardContent className="p-4">
              <div className="flex items-start justify-between">
                <div
                  className={cn(
                    'flex h-9 w-9 items-center justify-center rounded-lg bg-bg-secondary',
                    stat.color
                  )}
                >
                  <Icon className="h-4 w-4" />
                </div>
                {stat.trend && (
                  <TrendIcon
                    className={cn(
                      'h-3.5 w-3.5',
                      stat.trend === 'up' ? 'text-success' : stat.trend === 'down' ? 'text-danger' : 'text-muted'
                    )}
                  />
                )}
              </div>
              <p className="mt-3 text-2xl font-bold text-text">{stat.value}</p>
              <p className="mt-0.5 text-xs text-muted">{stat.label}</p>
              {stat.change && (
                <p className="mt-1.5 text-xs text-muted/80">{stat.change}</p>
              )}
            </CardContent>
          </Card>
        )
      })}
    </div>
  )
}
