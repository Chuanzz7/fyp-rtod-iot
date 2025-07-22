# !/usr/bin/env python3
"""
Optimized Picamera2 streaming service with FastAPI
Sends raw image data for backend processing with cv2.imdecode
"""
from __future__ import annotations
import queue
import threading
import time
import requests
from typing import Optional
import logging
import numpy as np
import cv2

from picamera2 import Picamera2
import libcamera
from fastapi import FastAPI, HTTPException, Response
import uvicorn

# Configuration
FASTAPI_URL = "http://192.168.0.214:8000/upload_frame"
VIDEO_SIZE = (640, 640)
TARGET_FPS = 15
SEND_THREADS = 4
MAX_QUEUE = 20  # Reduced since we're sending raw data
CAPTURE_INTERVAL = 1.0 / TARGET_FPS
SEND_TIMEOUT = 2.0

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Picamera2 Raw Frame Streamer")

# Global state
capture_thread: Optional[threading.Thread] = None
sender_threads: list[threading.Thread] = []
stop_event = threading.Event()
frame_queue: queue.Queue[bytes] = queue.Queue(maxsize=MAX_QUEUE)
session = requests.Session()


def init_camera() -> Picamera2:
    """Initialize and configure the camera for optimal performance"""
    try:
        picam2 = Picamera2()

        # Use video configuration for better performance
        config = picam2.create_video_configuration(
            main={"size": VIDEO_SIZE, "format": "RGB888"},
            controls={
                "FrameDurationLimits": (
                    int(1_000_000 // TARGET_FPS),
                    int(1_000_000 // TARGET_FPS)
                ),
                "AfMode": libcamera.controls.AfModeEnum.Continuous,
                "AeMeteringMode": libcamera.controls.AeMeteringModeEnum.Matrix,
                "AwbMode": libcamera.controls.AwbModeEnum.Auto,
                "Brightness": 0.0,
                "Contrast": 1.0,
            },
            buffer_count=4  # Buffer for smooth capture
        )

        picam2.configure(config)
        logger.info(f"Camera configured: {VIDEO_SIZE} @ {TARGET_FPS}fps, RGB888 format")
        return picam2

    except Exception as e:
        logger.error(f"Failed to initialize camera: {e}")
        raise


def encode_frame_to_jpeg(frame_array):
    # Convert RGB to BGR for OpenCV
    frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
    ret, jpeg = cv2.imencode('.jpg', frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if ret:
        return jpeg.tobytes()
    else:
        raise ValueError("Failed to encode frame to JPEG")


def sender_worker(worker_id: int):
    """Worker thread to send raw frame data to the API endpoint"""
    logger.info(f"Sender worker {worker_id} started")
    sent_count = 0

    while not stop_event.is_set():
        try:
            # Get frame with timeout
            frame_data = frame_queue.get(timeout=1.0)

            if frame_data is None:  # Sentinel to stop
                logger.info(f"Sender worker {worker_id} stopping (sent {sent_count} frames)")
                break

            # inside capture_worker loop after capturing frame_array:
            frame_array = np.frombuffer(frame_data, dtype=np.uint8).reshape((VIDEO_SIZE[1], VIDEO_SIZE[0], 3))
            frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
            ret, jpeg = cv2.imencode('.jpg', frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ret:
                frame_bytes = jpeg.tobytes()
                try:
                    response = session.post(
                        FASTAPI_URL,
                        data=frame_bytes,
                        headers={
                            "Content-Type": "application/octet-stream",
                            "X-Sent-Ts": str(time.time()),
                            "X-Worker-Id": str(worker_id),
                            "X-Frame-Shape": f"{VIDEO_SIZE[1]},{VIDEO_SIZE[0]},3",  # H,W,C
                            "X-Frame-Dtype": "uint8",
                            "Content-Length": str(len(frame_data))
                        },
                        timeout=SEND_TIMEOUT
                    )

                    if response.status_code == 200:
                        sent_count += 1
                    else:
                        logger.warning(f"API returned status {response.status_code}")
                except requests.exceptions.Timeout:
                    logger.warning(f"Send timeout (worker {worker_id})")
                    stop_stream()
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Send error (worker {worker_id}): {e}")
                except Exception as e:
                    logger.error(f"Unexpected error in sender {worker_id}: {e}")
            # enqueue frame_bytes
            else:
                logger.warning("Failed to encode frame")
            # Send frame to API



        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Critical error in sender worker {worker_id}: {e}")
            break
        finally:
            try:
                frame_queue.task_done()
            except ValueError:
                pass

    logger.info(f"Sender worker {worker_id} stopped (total sent: {sent_count})")


def capture_worker():
    """Main capture worker thread - captures raw RGB frames"""
    logger.info("Capture worker started")
    picam2 = None

    try:
        # Initialize camera
        picam2 = init_camera()
        picam2.start()
        logger.info("Camera started, warming up...")

        # Allow camera to warm up and stabilize
        time.sleep(2)

        frame_count = 0
        dropped_frames = 0
        start_time = time.time()
        last_capture_time = 0
        last_stats_time = start_time

        logger.info("Starting frame capture loop")

        # Main capture loop
        while not stop_event.is_set():
            current_time = time.time()

            # Rate limiting - maintain target FPS
            time_since_last = current_time - last_capture_time
            if time_since_last < CAPTURE_INTERVAL:
                sleep_time = CAPTURE_INTERVAL - time_since_last
                time.sleep(max(0.001, sleep_time))
                continue

            try:
                # Capture frame as numpy array (RGB888)
                frame_array = picam2.capture_array()

                if frame_array is not None and frame_array.size > 0:
                    # Convert numpy array to bytes for transmission
                    frame_bytes = frame_array.tobytes()

                    try:
                        frame_queue.put_nowait(frame_bytes)
                        frame_count += 1
                        last_capture_time = current_time

                    except queue.Full:
                        # Drop oldest frame if queue is full
                        try:
                            frame_queue.get_nowait()
                            frame_queue.put_nowait(frame_bytes)
                            frame_count += 1
                            dropped_frames += 1
                            last_capture_time = current_time
                        except queue.Empty:
                            dropped_frames += 1

                # Log stats every 10 seconds
                if current_time - last_stats_time >= 10.0:
                    elapsed = current_time - start_time
                    actual_fps = frame_count / elapsed if elapsed > 0 else 0
                    queue_util = (frame_queue.qsize() / MAX_QUEUE) * 100

                    logger.info(f"Stats - Frames: {frame_count}, Dropped: {dropped_frames}, "
                                f"FPS: {actual_fps:.1f}, Queue: {frame_queue.qsize()}/{MAX_QUEUE} ({queue_util:.1f}%)")
                    last_stats_time = current_time

            except Exception as e:
                logger.error(f"Error capturing frame: {e}")
                time.sleep(0.1)  # Brief pause on error

    except Exception as e:
        logger.error(f"Critical error in capture worker: {e}")
    finally:
        # Cleanup
        try:
            if picam2:
                picam2.stop()
                picam2.close()
                logger.info("Camera stopped and closed")
        except Exception as e:
            logger.error(f"Error during camera cleanup: {e}")

        # Send stop signals to sender threads
        for _ in range(SEND_THREADS):
            try:
                frame_queue.put_nowait(None)
            except queue.Full:
                # Force clear queue if needed
                try:
                    frame_queue.get_nowait()
                    frame_queue.put_nowait(None)
                except queue.Empty:
                    pass

        logger.info("Capture worker cleanup completed")


@app.post("/start_stream")
async def start_stream():
    """Start the video streaming"""
    global capture_thread, sender_threads

    if capture_thread and capture_thread.is_alive():
        return {
            "status": "already_running",
            "message": "Stream is already active",
            "queue_size": frame_queue.qsize()
        }

    try:
        # Clear any previous state
        stop_event.clear()
        sender_threads.clear()

        # Clear the queue
        cleared_frames = 0
        while not frame_queue.empty():
            try:
                frame_queue.get_nowait()
                cleared_frames += 1
            except queue.Empty:
                break

        if cleared_frames > 0:
            logger.info(f"Cleared {cleared_frames} old frames from queue")

        # Start sender threads first
        for i in range(SEND_THREADS):
            sender_thread = threading.Thread(
                target=sender_worker,
                args=(i,),
                name=f"SenderWorker-{i}",
                daemon=True
            )
            sender_threads.append(sender_thread)
            sender_thread.start()

        # Start capture thread
        capture_thread = threading.Thread(
            target=capture_worker,
            name="CaptureWorker",
            daemon=True
        )
        capture_thread.start()

        logger.info(f"Stream started: {SEND_THREADS} senders, target {TARGET_FPS}fps")

        return {
            "status": "started",
            "config": {
                "resolution": VIDEO_SIZE,
                "target_fps": TARGET_FPS,
                "sender_threads": SEND_THREADS,
                "max_queue_size": MAX_QUEUE,
                "format": "RGB888",
                "data_type": "raw_bytes"
            }
        }

    except Exception as e:
        logger.error(f"Failed to start stream: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start stream: {e}")


@app.post("/stop_stream")
async def stop_stream():
    """Stop the video streaming"""
    global capture_thread, sender_threads

    if not (capture_thread and capture_thread.is_alive()):
        return {"status": "not_running", "message": "Stream is not active"}

    try:
        logger.info("Stopping stream...")
        stop_start = time.time()

        # Signal all threads to stop
        stop_event.set()

        # Wait for capture thread with timeout
        if capture_thread:
            capture_thread.join(timeout=5.0)
            if capture_thread.is_alive():
                logger.warning("Capture thread didn't stop gracefully")
            capture_thread = None

        # Wait for all sender threads
        for i, thread in enumerate(sender_threads):
            thread.join(timeout=3.0)
            if thread.is_alive():
                logger.warning(f"Sender thread {i} didn't stop gracefully")

        sender_threads.clear()

        # Clear remaining frames
        frames_cleared = 0
        while not frame_queue.empty():
            try:
                frame_queue.get_nowait()
                frames_cleared += 1
            except queue.Empty:
                break

        stop_time = time.time() - stop_start
        logger.info(f"Stream stopped in {stop_time:.2f}s, cleared {frames_cleared} frames")

        return {
            "status": "stopped",
            "frames_cleared": frames_cleared,
            "stop_time_seconds": round(stop_time, 2)
        }

    except Exception as e:
        logger.error(f"Error stopping stream: {e}")
        raise HTTPException(status_code=500, detail=f"Error stopping stream: {e}")


@app.get("/status")
async def get_status():
    """Get current streaming status with performance metrics"""
    is_streaming = bool(capture_thread and capture_thread.is_alive())
    active_senders = len([t for t in sender_threads if t.is_alive()])

    return {
        "streaming": is_streaming,
        "performance": {
            "queue_size": frame_queue.qsize(),
            "max_queue_size": MAX_QUEUE,
            "queue_utilization_pct": round((frame_queue.qsize() / MAX_QUEUE) * 100, 1),
            "queue_full": frame_queue.full(),
            "active_senders": active_senders,
            "expected_senders": SEND_THREADS if is_streaming else 0
        },
        "config": {
            "resolution": VIDEO_SIZE,
            "target_fps": TARGET_FPS,
            "capture_interval_ms": round(CAPTURE_INTERVAL * 1000, 1),
            "api_endpoint": FASTAPI_URL,
            "send_timeout": SEND_TIMEOUT,
            "format": "RGB888_raw_bytes"
        }
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    is_streaming = bool(capture_thread and capture_thread.is_alive())

    return {
        "service": "Picamera2 Raw Frame Streamer",
        "version": "3.0",
        "healthy": True,
        "streaming": is_streaming,
        "timestamp": time.time(),
        "threads": {
            "capture_alive": is_streaming,
            "senders_alive": len([t for t in sender_threads if t.is_alive()]),
            "senders_configured": SEND_THREADS
        }
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Picamera2 Raw Frame Streamer",
        "status": "ready",
        "endpoints": ["/start_stream", "/stop_stream", "/status", "/health", "/photo"]
    }


@app.get("/photo")
async def shoot_photo():
    """
    Capture a single photo and return as JPEG image.
    """
    if capture_thread and capture_thread.is_alive():
        raise HTTPException(status_code=500, detail="Camera is Busy, Stop the Stream and Try again")

    try:
        picam2 = init_camera()
        picam2.start()
        time.sleep(1.0)  # Allow camera to warm up
        picam2.autofocus_cycle()
        frame_array = picam2.capture_array()
        picam2.stop()
        picam2.close()

        # Encode as JPEG
        frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
        ret, jpeg = cv2.imencode('.jpg', frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        if not ret:
            raise RuntimeError("JPEG encoding failed")
        return Response(content=jpeg.tobytes(), media_type="image/jpeg")
    except Exception as e:
        logger.error(f"Error shooting photo: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to capture photo: {e}")


# Graceful shutdown handler
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Service shutting down...")
    try:
        if capture_thread and capture_thread.is_alive():
            await stop_stream()
        session.close()
        logger.info("Shutdown completed")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


if __name__ == "__main__":
    try:
        logger.info("Starting Picamera2 Raw Frame Streamer v3.0")
        logger.info(f"Config: {VIDEO_SIZE} @ {TARGET_FPS}fps, {SEND_THREADS} senders")

        uvicorn.run(
            "__main__:app",  # This ensures it runs the current file
            host="0.0.0.0",
            port=9000,
            log_level="info",
            access_log=False  # Reduce noise
        )
    except KeyboardInterrupt:
        logger.info("Service interrupted by user")
    except Exception as e:
        logger.error(f"Failed to start service: {e}")

