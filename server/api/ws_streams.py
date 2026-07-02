import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Path
from server.inference.pipeline import inference_pipeline

router = APIRouter(tags=["Streams"])

class WebSocketConnection:
    """
    Wrapper around FastAPI WebSocket to handle thread-safe queueing
    and frame dropping to maintain real-time low latency.
    """
    def __init__(self, websocket: WebSocket, loop: asyncio.AbstractEventLoop):
        self.websocket = websocket
        self.loop = loop
        self.queue = asyncio.Queue(maxsize=15)
        # Self-referencing push_queue to match the broadcast implementation in pipeline.py
        self.push_queue = self

    def put_nowait(self, payload: dict):
        """Thread-safe push of a frame payload into the asyncio queue."""
        def _safe_enqueue():
            try:
                # Frame dropping mechanism to maintain real-time performance
                if self.queue.full():
                    try:
                        self.queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                self.queue.put_nowait(payload)
            except Exception as e:
                print(f"[WS Connection] Enqueue error: {e}")

        self.loop.call_soon_threadsafe(_safe_enqueue)

router = APIRouter(tags=["Streams"])

@router.websocket("/ws/streams/{camera_id}")
async def websocket_stream(
    websocket: WebSocket,
    camera_id: int = Path(..., description="Camera ID index (1-4)", ge=1, le=4)
):
    """
    WebSocket endpoint that streams annotated live frames (Base64 JPEG)
    and detection metadata for the specified camera.
    """
    await websocket.accept()
    loop = asyncio.get_running_loop()
    conn = WebSocketConnection(websocket, loop)

    # Register the connection with the pipeline
    inference_pipeline.register_websocket(camera_id, conn)

    try:
        while True:
            # Non-blocking fetch of the next queued frame payload
            payload = await conn.queue.get()
            await websocket.send_json(payload)
    except WebSocketDisconnect:
        # Expected disconnect when client navigates away
        pass
    except Exception as e:
        print(f"[WS CAM {camera_id}] Error in websocket session: {e}")
    finally:
        # Always clean up the registration on disconnect
        inference_pipeline.unregister_websocket(camera_id, conn)
