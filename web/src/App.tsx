import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { DashboardLayout } from '@/components/layout/DashboardLayout'
import { DashboardPage } from '@/pages/DashboardPage'
import { ZoneManagementPage } from '@/pages/ZoneManagementPage'
import { PlaceholderPage } from '@/pages/PlaceholderPage'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<DashboardLayout />}>
          <Route index element={<DashboardPage />} />
          <Route path="live-cameras" element={<PlaceholderPage title="Live Cameras" description="Full-screen multi-camera live view with AI overlays." />} />
          <Route path="cameras" element={<PlaceholderPage title="Camera Management" description="Add, configure, and monitor RTSP camera streams." />} />
          <Route path="zones" element={<ZoneManagementPage />} />
          <Route path="alerts" element={<PlaceholderPage title="Alerts" description="View and manage all AI-generated security alerts." />} />
          <Route path="events" element={<PlaceholderPage title="Event Logs" description="Searchable event history with filters and clips." />} />
          <Route path="attendance" element={<PlaceholderPage title="Attendance Reports" description="AI-based attendance tracking and reports." />} />
          <Route path="employees" element={<PlaceholderPage title="Employees / Users" description="Employee registration and access management." />} />
          <Route path="face-profiles" element={<PlaceholderPage title="Face Profiles" description="Secure face enrollment with consent tracking." />} />
          <Route path="alert-rules" element={<PlaceholderPage title="Alert Rules" description="Configure automated alert rules and notifications." />} />
          <Route path="access-control" element={<PlaceholderPage title="Access Control" description="Zone-based access control and violation logs." />} />
          <Route path="system-health" element={<PlaceholderPage title="System Health" description="Camera status, AI workers, and infrastructure monitoring." />} />
          <Route path="roles" element={<PlaceholderPage title="Roles & Permissions" description="Role-based access control matrix." />} />
          <Route path="settings" element={<PlaceholderPage title="Settings" description="Platform configuration and preferences." />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}

export default App
