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

### Download / Stream Recording Clip
* **Endpoint**: `GET /api/recordings/{alert_id}`
* **Path Parameters**:
  * `alert_id` (string, UUID): The unique alert ID from the alerts history.
* **Description**: Streams or downloads the 20-second MP4 video clip corresponding to the given alert event. Supports HTTP byte-range requests for interactive seeking in the web player.
* **Response Header**: `Content-Type: video/mp4`

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



