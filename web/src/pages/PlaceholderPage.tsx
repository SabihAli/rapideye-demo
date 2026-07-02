import { Card, CardContent } from '@/components/ui/Card'
import { Construction } from 'lucide-react'

interface PlaceholderPageProps {
  title: string
  description: string
}

export function PlaceholderPage({ title, description }: PlaceholderPageProps) {
  return (
    <div className="flex h-full items-center justify-center">
      <Card className="max-w-md text-center">
        <CardContent className="p-8">
          <Construction className="mx-auto h-12 w-12 text-muted" />
          <h1 className="mt-4 text-xl font-bold text-text">{title}</h1>
          <p className="mt-2 text-sm text-muted">{description}</p>
          <p className="mt-4 text-xs text-muted">
            This module will be built in upcoming iterations. Backend integration points are
            prepared for your partner&apos;s API.
          </p>
        </CardContent>
      </Card>
    </div>
  )
}
