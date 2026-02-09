# Configuration Guide

This project uses a central `.env` file in the project root for configuration.

## Key Configurations

| Variable | Default | Description |
|----------|---------|-------------|
| `MUSETALK_PORT` | `8001` | The port the server listens on. |
| `MUSETALK_GPU_ID` | `0` | The GPU ID to use. |
| `MUSETALK_BATCH_SIZE` | `4` | Inference batch size. Lower if OOM occurs. |
| `MUSETALK_FPS` | `25` | Output video FPS. |
| `MUSETALK_RESULT_DIR` | `./results` | Directory for generated videos. |

## Advanced Model Configuration

You can also point to custom model paths if needed:
- `MUSETALK_UNET_CONFIG`
- `MUSETALK_UNET_MODEL_PATH`
- `MUSETALK_WHISPER_DIR`
- `MUSETALK_VAE_TYPE`

## Running the Server

1. **Verify Environment**: Run `start_and_verify.bat` to check health.
2. **Start Server**: Run `start_server.bat` for normal operation.
