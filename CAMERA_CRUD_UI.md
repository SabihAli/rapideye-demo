# Camera Management — API Blueprint & UI Requirements

This document specifies the frontend work for the new dynamic camera
management feature. **The backend is already implemented** (see
`server/api/routes_cameras.py`, `server/inference/camera_registry.py`,
`server/schemas/cameras.py`, `tests/test_camera_registry.py`). This doc is
the contract for whoever builds the "Manage Cameras" UI and updates the
dashboard next.

## What changed, in one paragraph

The app used to hardcode exactly 4 cameras (`camera_1_url`..`camera_4_url`
in `server/config.py`, auto-started at boot). That's gone. The app now
starts with **zero cameras**. A user adds up to **4** via the new
`/api/cameras` endpoints (URL or uploaded video file), and can remove them.
Cameras persist in the existing SQLite DB (`data/rapideye_demo.db`) and
auto-start on every backend restart. Camera IDs are always an integer in
`[1, 4]` — a "slot" model, since zones/model-switches/recordings/alerts are
all already keyed by that same integer range. Adding a camera claims the
lowest free slot; removing frees it for reuse.

---

## Part 1 — API Blueprint

Base path: `/api/cameras`. All three endpoints are new.

### `GET /api/cameras`

Returns every currently-registered camera (0-4 items), with live stream
status merged in.

**Response `200`** — `CameraOut[]`:
```jsonc
[
  {
    "camera_id": 1,
    "name": "Front Lobby",
    "source_type": "url",           // "url" | "file"
    "source_value": "rtsp://192.168.1.50:554/stream1",
    "original_filename": null,       // only set when source_type == "file"
    "created_at": 1752470400.0,      // epoch seconds
    "is_active": true,               // decoder thread currently reading frames
    "current_fps": 29.7,
    "error_count": 0
  }
]
```
`source_value` for a `file`-type camera is a **server-side filesystem path**
(e.g. `data/videos/uploads/upload_3f1c...mp4`) — do not render it directly;
use `name`/`original_filename` for display.

```bash
curl http://localhost:8001/api/cameras
```

### `POST /api/cameras` — add a camera

Always `multipart/form-data`, regardless of source type — this keeps the
frontend on a single request shape.

| Field | Type | Required | Notes |
|---|---|---|---|
| `name` | text | yes | Display name. Non-blank after trimming. |
| `source_type` | text | yes | `"url"` or `"file"`. |
| `url` | text | only if `source_type=url` | Any string the decoder can open (RTSP/HTTP stream URL, or a server-local file path). Must NOT be sent together with `file`. |
| `file` | file | only if `source_type=file` | Must NOT be sent together with `url`. |

**Success — `201`**: returns the created `CameraOut` (same shape as above;
`is_active` will likely be `false` in the response itself since the decoder
thread has only just been asked to start — poll `GET /api/cameras` or the
websocket to see it flip to `true`).

**Errors:**

| Status | When | Body |
|---|---|---|
| `400` | `name` blank/whitespace-only; `source_type` is neither `url` nor `file`; `url` missing when `source_type=url`; `file` missing when `source_type=file`; both `url` and `file` provided | `{"detail": "<human-readable reason>"}` |
| `409` | 4 cameras already registered | `{"detail": "Maximum of 4 cameras already registered."}` |
| `415` | File extension not in the allowed set (`.mp4`, `.mov`, `.avi`, `.mkv` — case-insensitive) | `{"detail": "Unsupported file type '.xyz'. Allowed: ['.avi', '.mkv', '.mov', '.mp4']."}` |
| `413` | File exceeds 500MB — enforced on actual bytes streamed, not just the `Content-Length` header, so this can arrive **after** the upload has been streaming for a while on a slow connection | `{"detail": "File exceeds the 500MB limit."}` |
| `422` | Malformed multipart body (missing a required form field entirely — distinct from sending it empty) | FastAPI's default validation error shape |

```bash
# Add by URL
curl -X POST http://localhost:8001/api/cameras \
  -F "name=Front Lobby" -F "source_type=url" -F "url=rtsp://192.168.1.50:554/stream1"

# Add by file upload
curl -X POST http://localhost:8001/api/cameras \
  -F "name=Warehouse" -F "source_type=file" -F "file=@/path/to/clip.mp4;type=video/mp4"
```

### `DELETE /api/cameras/{camera_id}`

`camera_id` is a path int, `1 <= camera_id <= 4`.

Stops the camera's decoder thread immediately, deletes its zone
configuration and per-camera model-switch toggles (they reset to defaults —
see below), and deletes the uploaded video file on disk if it was a
`file`-type camera. **Past recordings and alert history for that camera_id
are left untouched** — they simply stop being associated with a live
stream; they are not deleted and do not need special UI handling here.

**Success — `204`**, empty body.

**Errors:**

| Status | When | Body |
|---|---|---|
| `404` | No camera registered at that `camera_id` | `{"detail": "Camera <id> is not registered."}` |

```bash
curl -X DELETE http://localhost:8001/api/cameras/1
```

### Interaction with existing endpoints

- `GET /api/health` and `GET /api/streams/status` — the `streams` array
  **no longer always has 4 entries**. It now has one entry per
  *currently-registered* camera (0-4). Any UI reading these (e.g. the
  dashboard's "N/4 streams active" counter) must compute its denominator
  from `GET /api/cameras`'s length, not assume 4.
- `GET/PUT /api/zones/{camera_id}`, `GET/PUT /api/cameras/{camera_id}/model-switches`,
  `/ws/streams/{camera_id}`, recordings endpoints — all unchanged in shape.
  They still accept any `camera_id` in `[1, 4]` regardless of whether a
  camera is currently registered there (this was already true before — an
  "unconfigured" camera_id just returns sensible defaults). In practice the
  frontend should only ever call these for `camera_id`s that `GET
  /api/cameras` actually returned.

---

## Part 2 — UI Requirements

### New "Manage Cameras" page

- Add a nav item in the sidebar (`web/src/components/layout/Sidebar.tsx` /
  `web/src/data/mockData.ts`'s `navItems`), e.g. `{ label: 'Manage Cameras',
  path: '/manage-cameras', icon: '<pick an available lucide icon>' }`, and
  wire the route in `web/src/App.tsx` alongside the other
  `<Route path="..." element={...} />` entries inside the `DashboardLayout`
  route group.
- **List view**: one card/row per registered camera (0-4), showing name,
  source type, live status (online/offline via `is_active`), current FPS,
  and a **Remove** button. Empty state (0 cameras): friendly prompt to add
  the first camera, no error styling.
- **Add flow**:
  1. Source-type toggle: **URL** vs **Local File** (radio/segmented
     control, not a dropdown — only 2 options, per the spec).
  2. **URL selected** → single text input for the stream URL. Basic
     client-side sanity check (non-empty; optionally warn if it doesn't
     look like `rtsp://`/`http(s)://`/a path — the backend doesn't
     validate URL reachability synchronously, it just starts a decoder
     thread that retries in the background, so don't block on a
     "test connection" step that doesn't exist).
  3. **Local File selected** → drag-and-drop zone **and** a "Browse" button
     (both must work — same file input underneath). On file selection,
     validate client-side *before* upload starts: extension in
     `.mp4/.mov/.avi/.mkv` (case-insensitive) and size ≤ 500MB. Show the
     specific violated constraint immediately — don't wait for the 415/413
     from the server if it's avoidable client-side. Show upload progress
     (the file can be up to 500MB; a progress bar, not just a spinner).
  4. Name field (required, non-blank after trim) — required regardless of
     source type.
  5. **Confirm** button — disabled until the current source-type's required
     fields are valid. On submit: disable the form, show a pending/loading
     state (this is a real network request with a file upload, not
     instant), then either show the new camera in the list (success) or a
     specific inline error mapped from the API's status code (see table
     below).
  6. The moment 4 cameras exist, disable/hide the "Add Camera" entry point
     entirely with a message explaining the 4-camera limit (don't just let
     them hit the 409 — surface the limit proactively once known via
     `GET /api/cameras`'s length).
- **Remove flow**: confirmation dialog before calling `DELETE` (destructive,
  irreversible — deletes zone config and model-switch settings for that
  slot). Clearly state in the confirmation copy that zone lines and
  detection toggles for this camera will be lost, but recordings/alert
  history will not.
- **Error → message mapping** for the add flow:

  | API status | Suggested user-facing message |
  |---|---|
  | `400` (blank name) | "Please enter a camera name." |
  | `400` (missing url/file) | "Please provide a {URL / file}." |
  | `409` | "You've reached the 4-camera limit. Remove a camera before adding another." |
  | `415` | "Unsupported file type. Use MP4, MOV, AVI, or MKV." |
  | `413` | "File is too large. Maximum size is 500MB." |
  | network/5xx | Generic retry-able error toast. |

- No rename/edit-in-place in this iteration — to change a camera's
  URL/file, remove it and re-add. (Explicitly out of scope, confirmed.)

### Dashboard changes

The dashboard currently assumes exactly 4 cameras via a **static** array:

- `web/src/lib/cameras.ts` — the `CAMERAS` constant must become a runtime
  fetch of `GET /api/cameras` (e.g. via a new hook/context, mirroring the
  existing `useDemo()`/`DemoContext` pattern already used for `health` and
  `alerts`) instead of a hardcoded list. `cameraNumId`/`cameraFromNumId`
  helpers should look up the fetched list instead of the static array.
- Every file currently importing `CAMERAS`/`CameraDefinition` from
  `web/src/lib/cameras.ts` needs to switch to the dynamic source:
  - `web/src/pages/DashboardPage.tsx` — camera grid + "N/4 streams active"
    counter (denominator becomes the fetched list's length, not a literal
    `4`).
  - `web/src/components/dashboard/CameraFeedCard.tsx` — consumes
    `CameraDefinition` as a prop type; no change needed to the type itself,
    just its source.
  - `web/src/components/dashboard/AlertFeed.tsx`
  - `web/src/context/ZoneContext.tsx`
  - `web/src/pages/RecordingsPage.tsx`
  - `web/src/pages/ZoneManagementPage.tsx`
  - `web/src/hooks/useStreamSocket.ts` (if it iterates the static list to
    open one websocket per camera — open one per *registered* camera
    instead, and handle a camera being added/removed while the dashboard is
    open by opening/closing sockets accordingly, not just on mount).
- **Empty state**: 0 cameras registered → the dashboard should not render 4
  "Offline" placeholder cards (that was the old hardcoded behavior). Show a
  message + link to the Manage Cameras page instead.
- **Responsive grid**: `DashboardPage.tsx`'s camera grid is currently
  `grid-cols-1 md:grid-cols-2 xl:grid-cols-4` sized around a fixed count of
  4. It must reflow correctly for 1, 2, 3, or 4 cameras (e.g. don't leave
  awkward gaps for 1-3 cameras at the `xl` breakpoint — consider capping
  columns at `min(count, 4)` rather than a fixed `xl:grid-cols-4`).

### Responsiveness (explicit requirement, both pages)

- Manage Cameras page and the Dashboard grid must both work at mobile,
  tablet, and desktop breakpoints (the app's existing Tailwind breakpoints:
  base/`md`/`xl`, per `DashboardPage.tsx`'s existing classes) — this was
  called out explicitly in the original feature request and isn't optional
  polish.
- The add-camera form (especially the drag-and-drop zone) needs a usable
  mobile fallback — drag-and-drop doesn't really exist on touch, so the
  "Browse" file-picker button must be the primary affordance on small
  screens, not a decorative secondary option next to the drop zone.
