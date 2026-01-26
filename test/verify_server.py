import httpx
import time
import os
import sys

SERVER_URL = "http://localhost:8000"
VIDEO_PATH = "test/media/short_test_video.mp4"
AUDIO_PATH = "test/media/test_audio.wav"
AVATAR_ID = "verify_test_avatar"

def log(msg, status="INFO"):
    print(f"[{status}] {msg}")

def check_health():
    log("Checking server health...")
    # Increase timeout to 5 minutes (150 * 2s) for first-time model loading
    for _ in range(150):
        try:
            r = httpx.get(f"{SERVER_URL}/health", timeout=5.0)
            if r.status_code == 200:
                data = r.json()
                if data["status"] == "running" and data["models"]["loaded"]:
                    log("Server is healthy and models loaded.", "SUCCESS")
                    return True
        except:
            pass
        time.sleep(2)
        if _ % 10 == 0:
            log("Waiting for server...", "WAIT")
    
    log("Server failed to start or load models.", "ERROR")
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

def verify_stream():
    log("Verifying streaming inference...")
    try:
        with open(AUDIO_PATH, "rb") as f:
            files = {"audio_file": (os.path.basename(AUDIO_PATH), f, "audio/wav")}
            
            start_t = time.time()
            frame_count = 0
            
            with httpx.stream("POST", f"{SERVER_URL}/inference/stream/{AVATAR_ID}", files=files, timeout=300) as r:
                if r.status_code != 200:
                    log(f"Stream connect failed: {r.status_code}", "ERROR")
                    return False
                    
                log("Stream connected. Receiving chunks...", "INFO")
                for chunk in r.iter_bytes():
                    if b'\xff\xd8' in chunk:
                        frame_count += 1
                        
            duration = time.time() - start_t
            if frame_count > 0:
                log(f"Stream verification successful. Received {frame_count} frames in {duration:.2f}s", "SUCCESS")
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
            
            r = httpx.post(f"{SERVER_URL}/inference/batch/{AVATAR_ID}", files=files, timeout=300)
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
