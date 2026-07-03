# RapidEye Security Demo - API Documentation

This document serves as the interface specification for frontend engineers building the React dashboard to integrate with the RapidEye Demo backend.

---

## General Configurations

* **Base URL**: `http://localhost:8000` (REST)
* **WebSocket URL**: `ws://localhost:8000` (WS)
* **CORS Policy**: Configured to allow all origins (`*`), methods, and headers for development convenience.

---

## 1. REST Endpoints

### Start / Reset Demo
* **Endpoint**: `POST /api/demo/start`
* **Description**: Starts or resets the security demo. Rewinds all hardcoded camera videos back to frame 0 and completely wipes the alert histories and active alert states from memory and disk.
* **Response Body (JSON)**:
  ```json
  {
    "status": "success",
    "message": "Demo started/reset successfully"
  }
  ```

### System Health
* **Endpoint**: `GET /api/health`
* **Description**: Returns the system status, CUDA GPU availability, a list of successfully loaded AI models on the GPU, and active decoding details for all 4 camera feeds.
* **Response Body (JSON)**:
  ```json
  {
    "gpu_available": true,
    "models_loaded": [
      "Entity (YOLO11)",
      "Fire/Smoke (YOLOv5)",
      "Weapons (YOLOv8)"
    ],
    "streams": [
      {
        "camera_id": 1,
        "is_active": true,
        "current_fps": 15.0,
        "error_count": 0
      },
      ...
    ]
  }
  ```

### Streams Status
* **Endpoint**: `GET /api/streams/status`
* **Description**: Returns the current FPS and error statuses for each of the 4 cameras.
* **Response Body (JSON)**:
  ```json
  [
    {
      "camera_id": 1,
      "is_active": true,
      "current_fps": 15.0,
      "error_count": 0
    },
    ...
  ]
  ```

### Zone Configuration CRUD

#### Get Zone Configuration
* **Endpoint**: `GET /api/zones/{camera_id}`
* **Path Parameters**:
  * `camera_id` (integer, 1-4)
* **Description**: Returns the configured polygon and alert classes for the specified camera.
* **Response Body (JSON)**:
  ```json
  {
    "camera_id": 1,
    "polygon": [
      [0.10, 0.10],
      [0.90, 0.10],
      [0.90, 0.90],
      [0.10, 0.90]
    ],
    "alert_classes": [
      "person",
      "vehicle",
      "fire",
      "smoke",
      "handgun",
      "rifle",
      "pistol",
      "knife"
    ]
  }
  ```

#### Update Zone Configuration
* **Endpoint**: `PUT /api/zones/{camera_id}`
* **Path Parameters**:
  * `camera_id` (integer, 1-4)
* **Description**: Updates or overwrites the zone configuration. Coordinates in the request body polygon list MUST be normalized (floats between `0.0` and `1.0`).
* **Request Body (JSON)**:
  ```json
  {
    "camera_id": 1,
    "polygon": [
      [0.2, 0.2],
      [0.8, 0.2],
      [0.8, 0.8],
      [0.2, 0.8]
    ],
    "alert_classes": ["person", "fire", "handgun"]
  }
  ```
* **Response Body (JSON)**: Returns the updated `ZoneConfig` object.

### Alerts Log History
* **Endpoint**: `GET /api/alerts`
* **Query Parameters**:
  * `limit` (integer, default `50`, min `1`, max `100`): Limit the size of records returned.
  * `offset` (integer, default `0`): Pagination offset.
* **Description**: Retrieves the alert events history logs in descending order (newest first).
* **Response Body (JSON)**:
  ```json
  [
    {
      "id": "e2646d60-705b-42fa-b7c1-cb3a9032cf6b",
      "camera_id": 1,
      "alert_type": "weapon.detected",
      "timestamp": 1782914102.45,
      "detections": [
        {
          "bbox": [0.45, 0.50, 0.55, 0.75],
          "class_name": "handgun",
          "confidence": 0.88
        }
      ],
      "clip_path": "/api/recordings/e2646d60-705b-42fa-b7c1-cb3a9032cf6b"
    }
  ]
  ```
  > **Note:** `clip_path` points to an alert clip playback URL. See [Recordings](#recordings) → Alert Clips.

### Recordings

The API supports two recording types:

| Type | Trigger | Storage | Metadata |
|------|---------|---------|----------|
| **Alert clips** | Automatic on zone/fire/weapon detection | `data/recordings/{alert_id}.mp4` | Alert event (`clip_path` on `GET /api/alerts`) |
| **Camera recordings** | Manual start/stop per camera | `data/recordings/cameras/camera_{n}/{id}.mp4` | SQLite (`data/rapideye_demo.db`) |

`POST /api/demo/start` stops any in-progress camera recordings and clears alert history (alert clip files on disk are not deleted).

---

#### Alert Clips

Alert clips are ~20-second MP4 files (10s pre-alert + 10s post-alert) written automatically when an alert fires. Each alert in `GET /api/alerts` includes a `clip_path` for playback once the clip has finished encoding.

##### Stream / Download Alert Clip
* **Endpoint**: `GET /api/recordings/{alert_id}`
* **Path Parameters**:
  * `alert_id` (string, UUID): The unique alert ID from the alerts history.
* **Description**: Streams or downloads the MP4 clip for the given alert event. Returns `404` if the clip file does not exist yet (encoding may still be in progress for ~10s after the alert).
* **Response Header**: `Content-Type: video/mp4`

---

#### Camera Recordings

Manual per-camera recordings save raw decoded frames from the live feed to disk. Metadata (start time, duration, status, playback URL) is stored in SQLite and exposed via the endpoints below.

##### Start Camera Recording
* **Endpoint**: `POST /api/cameras/{camera_id}/recordings/start`
* **Path Parameters**:
  * `camera_id` (integer, 1–4)
* **Description**: Begins saving live frames from the specified camera to an MP4 file on disk.
* **Response Body (JSON)**:
  ```json
  {
    "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "camera_id": 1,
    "started_at": 1783062000.12,
    "ended_at": null,
    "duration_seconds": null,
    "playback_url": "/api/camera-recordings/a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "file_size_bytes": null,
    "status": "recording"
  }
  ```
* **Errors**:
  * `409 Conflict` — camera is already recording.

##### Stop Camera Recording
* **Endpoint**: `POST /api/cameras/{camera_id}/recordings/stop`
* **Path Parameters**:
  * `camera_id` (integer, 1–4)
* **Description**: Stops the active recording for the camera, finalizes the MP4, and updates metadata (`ended_at`, `duration_seconds`, `file_size_bytes`, `status`).
* **Response Body (JSON)**: Same `CameraRecording` shape as start; `status` is `completed` or `failed` (failed if no frames were captured).
* **Errors**:
  * `409 Conflict` — camera is not currently recording.

##### Get Camera Recording Status
* **Endpoint**: `GET /api/cameras/{camera_id}/recordings/status`
* **Path Parameters**:
  * `camera_id` (integer, 1–4)
* **Description**: Returns whether the camera is actively recording and, if so, the in-progress recording metadata.
* **Response Body (JSON)**:
  ```json
  {
    "camera_id": 1,
    "is_recording": true,
    "recording": {
      "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "camera_id": 1,
      "started_at": 1783062000.12,
      "ended_at": null,
      "duration_seconds": null,
      "playback_url": "/api/camera-recordings/a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "file_size_bytes": null,
      "status": "recording"
    }
  }
  ```
  When not recording, `is_recording` is `false` and `recording` is `null`.

##### List Camera Recordings (per camera)
* **Endpoint**: `GET /api/cameras/{camera_id}/recordings`
* **Path Parameters**:
  * `camera_id` (integer, 1–4)
* **Query Parameters**:
  * `limit` (integer, default `50`, min `1`, max `100`)
  * `offset` (integer, default `0`)
* **Description**: Returns saved recordings for one camera, newest first.
* **Response Body (JSON)**: Array of `CameraRecording` objects (same shape as start/stop responses).

##### List All Camera Recordings
* **Endpoint**: `GET /api/camera-recordings`
* **Query Parameters**:
  * `limit` (integer, default `50`, min `1`, max `100`)
  * `offset` (integer, default `0`)
* **Description**: Returns saved recordings across all cameras, newest first.
* **Response Body (JSON)**: Array of `CameraRecording` objects.

##### Stream / Download Camera Recording
* **Endpoint**: `GET /api/camera-recordings/{recording_id}`
* **Path Parameters**:
  * `recording_id` (string, UUID): Recording ID from start/stop or list responses (`playback_url` is `/api/camera-recordings/{recording_id}`).
* **Description**: Streams or downloads the saved camera recording MP4.
* **Response Header**: `Content-Type: video/mp4`
* **Errors**:
  * `404 Not Found` — recording does not exist, file missing on disk, or status is `failed`.
  * `409 Conflict` — recording is still in progress (`status: recording`).

---

## 2. WebSockets

### Live Stream Endpoint
* **Endpoint**: `WS /ws/streams/{camera_id}`
* **Path Parameters**:
  * `camera_id` (integer, 1-4)
* **Description**: Pushes real-time JSON packets at the camera stream's active processing FPS. Each packet contains the annotated Base64 JPEG image and detection metadata (normalized bounding boxes, confidence, class labels).
* **Outgoing Message Payload (JSON)**:
  ```json
  {
    "camera_id": 1,
    "frame": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD...",
    "detections": [
      {
        "bbox": [0.12, 0.24, 0.34, 0.67],
        "class_name": "person",
        "confidence": 0.91
      }
    ],
    "is_alert": false,
    "target_fps": 15.0,
    "latency_ms": 14.5
  }
  ```

* **Frontend Optimization Tip**: To achieve smooth playback:
  1. Open a WebSocket connection for each active camera grid tile.
  2. Bind the `"frame"` string directly to the `src` attribute of a standard HTML `<img>` tag.
  3. Render the detections overlay and coordinates if doing client-side calculations, though the frame is pre-annotated on the backend (with color-coded boxes and zones) to ensure zero sync lag.



