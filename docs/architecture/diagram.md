# Pipeline and Data Flow Architecture

This document details the visual and logical relationships between the **Stream Ingestion**, **Adaptive GPU Inference Pipeline**, and the **React Dashboard** as specified in the [PLAN.md](file:///c:/Programming/Projects/03_PROTOTYPE/rapideye_prototype/rapideye-demo/PLAN.md).

## Architecture Diagram

```mermaid
flowchart TB
    %% ---------------------------------------
    %% STYLES & INTERFACES
    %% ---------------------------------------
    classDef inputStyle fill:#2A2D34,stroke:#4E5D6C,stroke-width:2px,color:#FFFFFF;
    classDef backendStyle fill:#1E293B,stroke:#38BDF8,stroke-width:2px,color:#F8FAFC;
    classDef inferenceStyle fill:#14532D,stroke:#4ADE80,stroke-width:2px,color:#F0FDF4;
    classDef frontendStyle fill:#311B92,stroke:#B388FF,stroke-width:2px,color:#EDE7F6;
    classDef storageStyle fill:#78350F,stroke:#F59E0B,stroke-width:2px,color:#FEF3C7;

    %% ---------------------------------------
    %% INPUT LAYER
    %% ---------------------------------------
    subgraph Inputs ["Stream Ingestion (4 Independent Feeds)"]
        YT["YouTube URL / RTSP / HLS"]:::inputStyle
        FF["FFmpeg Demuxer (Independent Threads)"]:::inputStyle
        YT --> FF
    end

    %% ---------------------------------------
    %% BACKEND SERVICE (FastAPI)
    %% ---------------------------------------
    subgraph Backend ["FastAPI Server Backend"]
        direction TB

        subgraph IngestPipe ["Ingestion & Scheduling"]
            Buf["Per-Camera Ring Buffer (10s Pre-Alert Deque)"]:::backendStyle
            Sched["Adaptive FPS Scheduler (Rebalances FPS based on activity)"]:::backendStyle
        end

        subgraph InfPipeline ["GPU Inference Pipeline (RTX 3090)"]
            direction LR
            YOLO_Ent["YOLO Entity Detection (COCO Classes)"]:::inferenceStyle
            YOLO_Fire["YOLO Fire & Smoke Detection (Custom Weights)"]:::inferenceStyle
            YOLO_Weap["YOLO Weapon Detection (Custom Weights)"]:::inferenceStyle
            Merge["Unified Result Merger"]:::inferenceStyle

            YOLO_Ent --> Merge
            YOLO_Fire --> Merge
            YOLO_Weap --> Merge
        end

        subgraph PostProcess ["Post-Processing & Storage"]
            Zone["Zone Engine (Point-in-polygon intrusion checks)"]:::backendStyle
            Clip["Alert Clip Writer (10s Pre + 10s Post MP4)"]:::backendStyle
            Annotator["Frame Annotator (Bboxes, labels, zone overlays)"]:::backendStyle
        end

        %% Connections within Backend
        FF -->|"Raw Frames (~15 FPS)"| Buf
        Buf --> Sched
        Sched -->|"Adaptive Target FPS"| InfPipeline
        
        Merge -->|"Detections Payload"| Zone
        Zone -->|"Alert Event"| Clip
        Merge -->|"Bboxes"| Annotator
    end

    %% ---------------------------------------
    %% STORAGE LAYER
    %% ---------------------------------------
    subgraph Storage ["Local Storage (gitignored)"]
        ModelsDB["data/models/"]:::storageStyle
        ZonesDB["data/zones/*.json"]:::storageStyle
        RecordingsDB["data/recordings/*.mp4"]:::storageStyle
    end

    %% Model files loading
    ModelsDB -.->|"Loads weights"| InfPipeline
    ZonesDB -.->|"Reads/Writes polygons"| Zone
    Clip -->|"Writes MP4 clips"| RecordingsDB

    %% ---------------------------------------
    %% API / TRANSPORT LAYER
    %% ---------------------------------------
    subgraph Transport ["API & WebSocket Interfaces"]
        WS_Stream["WebSocket: /ws/streams/{camera_id}\n(Annotated JPEG + Bbox Metadata)"]:::backendStyle
        REST_API["REST API:\n- GET /api/alerts\n- GET/PUT /api/zones/{camera_id}\n- GET /api/recordings/{alert_id}"]:::backendStyle
    end

    Annotator -->|"Annotated Frames"| WS_Stream
    REST_API -.->|"Queries / Mutations"| Zone
    REST_API -.->|"Downloads clips"| RecordingsDB

    %% ---------------------------------------
    %% FRONTEND CLIENT (React)
    %% ---------------------------------------
    subgraph Frontend ["React Web Dashboard"]
        Grid["Camera Grid (4 Live Tiles)"]:::frontendStyle
        ZEditor["Zone Polygon Editor"]:::frontendStyle
        ALog["Alerts Log Panel"]:::frontendStyle
    end

    %% API Connections to Frontend
    WS_Stream -->|"JPEG Binaries & JSON Metadata"| Grid
    REST_API <-->|"Zone configs CRUD"| ZEditor
    REST_API -->|"Alert list & video player"| ALog
```

## Step-by-Step Flow Explanation

1. **Ingest Phase:** `FFmpeg` demuxes each input stream (YouTube, RTSP, or HLS) on its own separate thread. Raw frames are stored in a **10-second rolling RAM ring buffer** to capture pre-alert footage.
2. **Scheduling Phase:** The **Adaptive FPS Scheduler** looks at GPU load and recent detections to adjust the FPS budget dynamically across all 4 cameras.
3. **Inference Pipeline:** Frames are passed to the GPU (RTX 3090) to run through three YOLO models:
   * **Entity Model:** Detects humans, cars, animals, etc.
   * **Fire Model:** Detects active fire and smoke (loaded via `torch.hub`).
   * **Weapon Model:** Detects pistols, rifles, knives, etc.
4. **Result Merging & Zone Analysis:** The detections are merged. The **Zone Engine** tests if class centroids lie inside the user-configured polygon. If a person, fire, or weapon triggers the zone boundary:
   * An alert is fired.
   * **Alert Clip Writer** extracts the pre-alert frames from the ring buffer, captures the next 10 seconds of post-alert footage, and writes a `.mp4` file to local storage.
5. **Annotation & WebSocket Stream:** The **Frame Annotator** renders color-coded bounding boxes (e.g., orange for fire, red for weapons, green for safe zones) and writes the output frame to a JPEG binary streamed live over WebSockets to the React frontend.
6. **Frontend Experience:** The React client displays a 2x2 grid updating at target FPS, lets administrators draw zone polygons via the **Zone Editor**, and alerts them instantly with clip-playback links in the **Event Log**.
