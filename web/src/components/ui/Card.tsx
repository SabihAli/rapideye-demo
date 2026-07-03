import { cn } from '@/lib/utils'

interface CardProps {
  children: React.ReactNode
  className?: string
}

export function Card({ children, className }: CardProps) {
  return (
    <div
      className={cn(
        'rounded-2xl border border-white/5 bg-bg-card',
        className
      )}
    >
      {children}
    </div>
  )
}

export function CardHeader({ children, className }: CardProps) {
  return (
    <div className={cn('flex items-center justify-between p-4 pb-0', className)}>
      {children}
    </div>
  )
}

export function CardContent({ children, className }: CardProps) {
  return <div className={cn('p-4', className)}>{children}</div>
}

export function CardTitle({ children, className }: CardProps) {
  return (
    <h3 className={cn('text-sm font-semibold text-text', className)}>
      {children}
    </h3>
  )
}
