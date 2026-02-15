import httpx
import time
import os
import sys

def load_port_from_env_file(env_path):
    if not os.path.exists(env_path):
        return None

    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key.strip() == "MUSETALK_PORT":
                    port = value.strip().strip('"').strip("'")
                    return port
    except Exception:
        return None

    return None

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENV_PORT = load_port_from_env_file(os.path.join(ROOT_DIR, ".env"))
PORT = os.getenv("MUSETALK_PORT") or ENV_PORT or "8000"
SERVER_URL = f"http://localhost:{PORT}"
VIDEO_PATH = "test/media/short_test_video.mp4"
AUDIO_PATH = "test/media/test_audio.wav"
AVATAR_ID = "verify_test_avatar"

def log(msg, status="INFO"):
    print(f"[{status}] {msg}")

VERIFY_STARTUP_TIMEOUT = float(os.getenv("VERIFY_STARTUP_TIMEOUT", "180"))
VERIFY_REQUIRE_MODELS = os.getenv("VERIFY_REQUIRE_MODELS", "1").lower() not in {"0", "false", "no"}

def check_health():
    log("Checking server health...")
    start_time = time.time()
    attempts = 0

    while (time.time() - start_time) < VERIFY_STARTUP_TIMEOUT:
        attempts += 1
        try:
            r = httpx.get(f"{SERVER_URL}/health", timeout=5.0)
            if r.status_code == 200:
                data = r.json()
                if data.get("status") == "running":
                    if not VERIFY_REQUIRE_MODELS or data.get("models", {}).get("loaded"):
                        log("Server is healthy and models loaded.", "SUCCESS")
                        return True
        except Exception:
            pass

        time.sleep(2)
        if attempts % 10 == 0:
            log("Waiting for server...", "WAIT")

    if VERIFY_REQUIRE_MODELS:
        log("Server failed to start or load models.", "ERROR")
    else:
        log("Server did not report models loaded before timeout.", "ERROR")
    return False

def preprocess():
    log(f"Preprocessing avatar '{AVATAR_ID}'...")
    if not os.path.exists(VIDEO_PATH):
        log(f"Video file missing: {VIDEO_PATH}", "ERROR")
        return False
        
    try:
        with open(VIDEO_PATH, "rb") as f:
            files = {"video_file": (os.path.basename(VIDEO_PATH), f, "video/mp4")}
            data = {"avatar_id": AVATAR_ID, "bbox_shift": 0}
            
            r = httpx.post(f"{SERVER_URL}/avatars/preprocess", data=data, files=files, timeout=300)
            if r.status_code == 200:
                log("Avatar preprocessed successfully.", "SUCCESS")
                return True
            else:
                log(f"Preprocess failed: {r.text}", "ERROR")
                return False
    except Exception as e:
        log(f"Preprocess exception: {e}", "ERROR")
        return False

STREAM_FRAME_TARGET = int(os.getenv("VERIFY_STREAM_FRAMES", "5"))
STREAM_TIMEOUT = float(os.getenv("VERIFY_STREAM_TIMEOUT", "900"))
BATCH_TIMEOUT = float(os.getenv("VERIFY_BATCH_TIMEOUT", "300"))

def verify_stream():
    log("Verifying streaming inference...")
    try:
        with open(AUDIO_PATH, "rb") as f:
            files = {"audio_file": (os.path.basename(AUDIO_PATH), f, "audio/wav")}
            
            start_t = time.time()
            frame_count = 0
            
            with httpx.stream(
                "POST",
                f"{SERVER_URL}/inference/stream/{AVATAR_ID}",
                files=files,
                timeout=STREAM_TIMEOUT
            ) as r:
                if r.status_code != 200:
                    log(f"Stream connect failed: {r.status_code}", "ERROR")
                    return False
                    
                log("Stream connected. Receiving chunks...", "INFO")
                for chunk in r.iter_bytes():
                    if b'\xff\xd8' in chunk:
                        frame_count += 1
                        if frame_count >= STREAM_FRAME_TARGET:
                            break
                        
            duration = time.time() - start_t
            if frame_count > 0:
                log(
                    f"Stream verification successful. Received {frame_count} frames in {duration:.2f}s",
                    "SUCCESS"
                )
                return True
            else:
                log("Stream received 0 frames.", "ERROR")
                return False
                
    except Exception as e:
        log(f"Stream verification exception: {e}", "ERROR")
        return False

def verify_batch():
    log("Verifying batch inference...")
    try:
        with open(AUDIO_PATH, "rb") as f:
            files = {"audio_file": (os.path.basename(AUDIO_PATH), f, "audio/wav")}
            
            r = httpx.post(
                f"{SERVER_URL}/inference/batch/{AVATAR_ID}",
                files=files,
                timeout=BATCH_TIMEOUT
            )
            if r.status_code == 200:
                content_type = r.headers.get("content-type")
                size = len(r.content)
                log(f"Batch response: {content_type}, Size: {size} bytes", "INFO")
                
                if size > 1000 and "video/mp4" in content_type:
                    output_file = "test/result.mp4"
                    with open(output_file, "wb") as out:
                        out.write(r.content)
                    log(f"Batch verification successful. Saved to {output_file}", "SUCCESS")
                    return True
                else:
                    log("Batch response invalid (small size or wrong type).", "ERROR")
                    return False
            else:
                log(f"Batch failed: {r.status_code} - {r.text}", "ERROR")
                return False
    except Exception as e:
        log(f"Batch verification exception: {e}", "ERROR")
        return False

if __name__ == "__main__":
    if not check_health():
        sys.exit(1)
        
    if not preprocess():
        sys.exit(1)
        
    stream_ok = verify_stream()
    # stream_ok = True # Skip streaming to test batch generation
    batch_ok = verify_batch()
    
    if stream_ok and batch_ok:
        log("ALL VERIFICATION STEPS PASSED", "SUCCESS")
        sys.exit(0)
    else:
        log("Verification failed", "ERROR")
        sys.exit(1)
