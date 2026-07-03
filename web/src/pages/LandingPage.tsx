import { Link } from 'react-router-dom'
import { Flame, ScanFace, Shield, Camera, MapPin, Bell, ArrowRight } from 'lucide-react'
import { Button } from '@/components/ui/Button'

const features = [
  {
    icon: Camera,
    title: 'Live Multi-Camera Feeds',
    description: 'Monitor four RTSP streams in a unified grid with real-time AI overlays and stream health.',
  },
  {
    icon: MapPin,
    title: 'Zone Management',
    description: 'Draw restricted zones on live frames and trigger alerts when objects enter protected areas.',
  },
  {
    icon: Bell,
    title: 'Live Alert Feed',
    description: 'Review detections as they happen with timestamps, confidence scores, and clip playback.',
  },
  {
    icon: Shield,
    title: 'Enterprise Command Center',
    description: 'Built for 24/7 security operations with a dark, high-contrast monitoring interface.',
  },
]

const models = [
  { icon: Flame, name: 'Fire Detection', detail: 'Smoke and flame recognition across all camera feeds.' },
  { icon: ScanFace, name: 'Face Recognition', detail: 'Identify enrolled profiles with consent-aware enrollment.' },
  { icon: Shield, name: 'Weapon Detection', detail: 'Flag potential weapons in real time for rapid response.' },
]

export function LandingPage() {
  return (
    <div className="min-h-screen bg-bg-main text-text-secondary">
      <header className="border-b border-white/5 bg-bg-card">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
          <p className="text-lg font-semibold tracking-tight text-text">RapidEye</p>
          <div className="flex items-center gap-2">
            <Link to="/dashboard">
              <Button variant="outline" size="sm">
                Sign In
              </Button>
            </Link>
            <Link to="/dashboard">
              <Button size="sm">
                Open Dashboard
                <ArrowRight className="h-3.5 w-3.5" />
              </Button>
            </Link>
          </div>
        </div>
      </header>

      <section className="mx-auto max-w-6xl px-6 pb-16 pt-14 md:pt-20">
        <div className="grid items-center gap-12 lg:grid-cols-2">
          <div>
            <p className="mb-3 text-xs font-medium uppercase tracking-widest text-muted">
              AI Video Intelligence
            </p>
            <h1 className="text-4xl font-bold leading-tight text-text md:text-5xl">
              Turn your camera network into real-time security decisions.
            </h1>
            <p className="mt-5 max-w-lg text-base leading-relaxed text-muted">
              RapidEye is an enterprise surveillance command center that combines live feeds, zone-based
              access control, and AI detection for fire, faces, and weapons in one unified dashboard.
            </p>
            <div className="mt-8 flex flex-wrap gap-3">
              <Link to="/dashboard">
                <Button size="lg">Get Started</Button>
              </Link>
              <Link to="/zones">
                <Button variant="outline" size="lg">
                  Configure Zones
                </Button>
              </Link>
            </div>
          </div>

          <div className="rounded-3xl border border-white/10 bg-bg-card p-4 shadow-2xl">
            <div className="mb-3 flex items-center justify-between px-1">
              <p className="text-sm font-medium text-text">Security Monitor</p>
              <span className="rounded-full border border-white/10 px-2.5 py-0.5 text-[10px] text-muted">
                4 streams
              </span>
            </div>
            <div className="grid grid-cols-2 gap-2">
              {['Camera 1', 'Camera 2', 'Camera 3', 'Camera 4'].map((name) => (
                <div
                  key={name}
                  className="aspect-video rounded-2xl border border-white/5 bg-bg-secondary p-2"
                >
                  <div className="flex items-center justify-between">
                    <span className="rounded-full bg-black/70 px-2 py-0.5 text-[9px] text-danger">LIVE</span>
                    <span className="text-[9px] text-muted">AI</span>
                  </div>
                  <div className="mt-6 flex justify-center gap-1">
                    {['Fire', 'Face', 'Weapon'].map((tag) => (
                      <span
                        key={tag}
                        className="rounded-full border border-white/15 bg-black/50 px-2 py-0.5 text-[8px] text-text"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                  <p className="mt-3 text-[10px] text-text">{name}</p>
                </div>
              ))}
            </div>
            <div className="mt-3 rounded-2xl border border-dashed border-white/10 px-4 py-6 text-center text-xs text-muted">
              Live alerts and detection events stream here in real time
            </div>
          </div>
        </div>
      </section>

      <section className="border-y border-white/5 bg-bg-card">
        <div className="mx-auto max-w-6xl px-6 py-14">
          <h2 className="text-2xl font-bold text-text">Built for modern security operations</h2>
          <p className="mt-2 max-w-2xl text-sm text-muted">
            Everything operators need in a single pane of glass: live video, intelligent alerts, and
            configurable zones without switching tools.
          </p>
          <div className="mt-8 grid gap-4 sm:grid-cols-2">
            {features.map(({ icon: Icon, title, description }) => (
              <div
                key={title}
                className="rounded-2xl border border-white/5 bg-bg-secondary p-5 transition-colors hover:border-white/10"
              >
                <div className="mb-3 flex h-9 w-9 items-center justify-center rounded-full border border-white/10 bg-bg-card">
                  <Icon className="h-4 w-4 text-text" />
                </div>
                <h3 className="text-base font-semibold text-text">{title}</h3>
                <p className="mt-2 text-sm leading-relaxed text-muted">{description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-6 py-14">
        <h2 className="text-2xl font-bold text-text">AI detection models</h2>
        <p className="mt-2 text-sm text-muted">
          Toggle detection modes per camera directly from the live feed control bar.
        </p>
        <div className="mt-8 grid gap-4 md:grid-cols-3">
          {models.map(({ icon: Icon, name, detail }) => (
            <div
              key={name}
              className="rounded-2xl border border-white/5 bg-bg-card p-5"
            >
              <Icon className="mb-3 h-5 w-5 text-text" />
              <h3 className="font-semibold text-text">{name}</h3>
              <p className="mt-2 text-sm text-muted">{detail}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="border-t border-white/5 bg-bg-card">
        <div className="mx-auto flex max-w-6xl flex-col items-start justify-between gap-6 px-6 py-12 md:flex-row md:items-center">
          <div>
            <h2 className="text-xl font-bold text-text">Ready to monitor your facility?</h2>
            <p className="mt-1 text-sm text-muted">Launch the dashboard and connect your camera streams.</p>
          </div>
          <Link to="/dashboard">
            <Button size="lg">
              Open Dashboard
              <ArrowRight className="h-4 w-4" />
            </Button>
          </Link>
        </div>
      </section>
    </div>
  )
}
