# Memory Optimization Guide

## Problem
`RuntimeError: DefaultCPUAllocator: not enough memory: you tried to allocate 117964800 bytes.`

This error occurs when the system runs out of RAM during model inference.

## Applied Fixes

### 1. ✅ Reduced Batch Size (conf.py)
- **Changed**: `batch_size` from 20 → 4
- **Impact**: Reduces memory usage by ~80% during inference
- **Trade-off**: Slightly slower processing

### 2. ✅ Added Memory Cleanup (model_loader.py)
- Added `torch.cuda.empty_cache()` and `gc.collect()` after model loading
- Clears unused memory between model initialization steps

### 3. ✅ Added Inference Memory Management (inference.py)
- Delete tensors after each batch processing
- Clear CUDA cache after prediction worker completes
- Clear memory after streaming completes

## Additional Recommendations

### Environment Variables
Set these before running the server to limit memory usage:

```bash
# For Windows PowerShell
$env:PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
$env:OMP_NUM_THREADS="4"

# For Linux/Mac
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export OMP_NUM_THREADS=4
```

### Further Reduce Batch Size If Needed
Edit `musetalk_server/conf.py`:
```python
batch_size: int = 2  # Even smaller if still OOM
```

### Use Environment Variable Override
```bash
# Windows PowerShell
$env:MUSETALK_BATCH_SIZE="2"
python -m musetalk_server.app

# Linux/Mac
MUSETALK_BATCH_SIZE=2 python -m musetalk_server.app
```

### Close Other Applications
- Close Chrome/Edge browsers (they use a lot of RAM)
- Close other Python processes
- Close IDEs if running the server from terminal

### System Recommendations
- **Minimum RAM**: 16 GB
- **Recommended RAM**: 32 GB
- **With GPU**: Can reduce CPU RAM requirements

### Monitor Memory Usage
```bash
# Windows
tasklist /FI "IMAGENAME eq python.exe" /V

# Linux/Mac
htop
```

## Verification
After applying fixes, restart the server:
```bash
cd MuseTalk
conda activate MuseTalk
python -m musetalk_server.app
```

Test with a small audio file first to verify the memory issue is resolved.
