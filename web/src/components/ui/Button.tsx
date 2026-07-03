import { cn } from '@/lib/utils'

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'default' | 'outline' | 'ghost' | 'danger'
  size?: 'sm' | 'md' | 'lg'
}

const variants = {
  default: 'bg-primary text-inverse hover:bg-text-secondary font-medium',
  outline: 'border border-white/20 bg-transparent text-text hover:bg-white/5',
  ghost: 'bg-transparent text-muted hover:bg-bg-secondary hover:text-text',
  danger: 'border border-danger/40 bg-transparent text-danger hover:bg-danger/10',
}

const sizes = {
  sm: 'px-3 py-1 text-xs rounded-full',
  md: 'px-5 py-2 text-sm rounded-full',
  lg: 'px-6 py-2.5 text-sm rounded-full',
}

export function Button({
  children,
  variant = 'default',
  size = 'md',
  className,
  type = 'button',
  ...props
}: ButtonProps) {
  return (
    <button
      type={type}
      className={cn(
        'inline-flex items-center justify-center gap-2 transition-colors duration-150 disabled:opacity-40 disabled:cursor-not-allowed',
        variants[variant],
        sizes[size],
        className
      )}
      {...props}
    >
      {children}
    </button>
  )
}
