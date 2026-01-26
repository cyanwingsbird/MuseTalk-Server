import httpx
import time
import os

# Configuration
SERVER_URL = "http://localhost:8000"
TEST_VIDEO = "data/video/short.mp4" # Replace with actual path if running
TEST_AUDIO = "data/audio/eng.wav" # Replace with actual path if running
AVATAR_ID = "short_avatar"

def check_health():
    print(f"Checking health at {SERVER_URL}/health...")
    try:
        response = httpx.get(f"{SERVER_URL}/health")
        response.raise_for_status()
        print("Health Check:", response.json())
        return True
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def preprocess_avatar():
    print(f"\nPreprocessing avatar '{AVATAR_ID}'...")
    if not os.path.exists(TEST_VIDEO):
        print(f"Skipping preprocess: Test video not found at {TEST_VIDEO}")
        return

    try:
        with open(TEST_VIDEO, "rb") as f:
            files = {"video_file": (os.path.basename(TEST_VIDEO), f, "video/mp4")}
            data = {"avatar_id": AVATAR_ID, "bbox_shift": 0}
            
            # This might take a while depending on video length
            response = httpx.post(
                f"{SERVER_URL}/avatars/preprocess", 
                data=data, 
                files=files, 
                timeout=None # Disable timeout for long processing
            )
            response.raise_for_status()
            print("Preprocess Result:", response.json())
    except Exception as e:
        print(f"Preprocess failed: {e}")

def stream_inference():
    print(f"\nStarting streaming inference for '{AVATAR_ID}'...")
    if not os.path.exists(TEST_AUDIO):
        print(f"Skipping stream: Test audio not found at {TEST_AUDIO}")
        return

    try:
        with open(TEST_AUDIO, "rb") as f:
            files = {"audio_file": (os.path.basename(TEST_AUDIO), f, "audio/wav")}
            
            # Use a streaming request
            with httpx.stream("POST", f"{SERVER_URL}/inference/stream/{AVATAR_ID}", files=files, timeout=None) as response:
                response.raise_for_status()
                
                print("Stream connected. Receiving frames...")
                frame_count = 0
                bytes_received = 0
                start_time = time.time()
                
                # Consume the multipart stream
                # In a real app, you'd parse the boundary and extract JPEG frames
                # Here we just count bytes to verify data flow
                for chunk in response.iter_bytes():
                    bytes_received += len(chunk)
                    # Primitive frame counting heuristic (searching for JPEG header)
                    if b'\xff\xd8' in chunk:
                        frame_count += 1
                        print(f"Received frame {frame_count}", end='\r')
                
                duration = time.time() - start_time
                print(f"\nStream finished. Received {bytes_received / 1024 / 1024:.2f} MB in {duration:.2f}s")
                
    except Exception as e:
        print(f"Streaming failed: {e}")

def batch_inference():
    print(f"\nStarting batch inference for '{AVATAR_ID}'...")
    if not os.path.exists(TEST_AUDIO):
        print(f"Skipping batch: Test audio not found at {TEST_AUDIO}")
        return

    try:
        with open(TEST_AUDIO, "rb") as f:
            files = {"audio_file": (os.path.basename(TEST_AUDIO), f, "audio/wav")}
            
            response = httpx.post(
                f"{SERVER_URL}/inference/batch/{AVATAR_ID}", 
                files=files, 
                timeout=None
            )
            response.raise_for_status()
            
            output_file = f"result_{AVATAR_ID}.mp4"
            with open(output_file, "wb") as out:
                out.write(response.content)
            
            print(f"Batch inference complete. Saved to {output_file}")
            
    except Exception as e:
        print(f"Batch inference failed: {e}")

if __name__ == "__main__":
    if check_health():
        # Uncomment to run actual processing if files exist
        preprocess_avatar() 
        stream_inference()
        # batch_inference()
        pass
