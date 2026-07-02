import { cn } from '@/lib/utils'

interface BadgeProps {
  children: React.ReactNode
  variant?: 'default' | 'success' | 'warning' | 'danger' | 'info' | 'outline'
  className?: string
}

const variants = {
  default: 'bg-bg-secondary text-text-secondary border-border',
  success: 'bg-bg-elevated text-success border-border',
  warning: 'bg-bg-elevated text-warning border-border',
  danger: 'bg-bg-elevated text-danger border-border',
  info: 'bg-bg-secondary text-text-secondary border-border',
  outline: 'bg-transparent text-muted border-border',
}

export function Badge({ children, variant = 'default', className }: BadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-medium',
        variants[variant],
        className
      )}
    >
      {children}
    </span>
  )
}
