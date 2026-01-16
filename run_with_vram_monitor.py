"""Run sam-mosaic with VRAM monitoring."""
import subprocess
import threading
import time
import json
from datetime import datetime

# Configuration
INPUT_IMAGE = r"D:\cubo_2002_NDVI_mnf_3bandas\cubo_2002_NDVI_mnf_3bandas"
OUTPUT_DIR = r"D:\cubo_2002_NDVI_mnf_3bandas\output"
CHECKPOINT = r"D:\sam-mosaic2\checkpoints\sam2.1_hiera_large.pt"
VRAM_LOG = r"D:\cubo_2002_NDVI_mnf_3bandas\vram_log.json"
MONITOR_INTERVAL = 10  # seconds

# VRAM monitoring
vram_data = []
stop_monitoring = False

def get_vram():
    """Get VRAM usage via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            used, total = map(int, result.stdout.strip().split(", "))
            return used, total
    except:
        pass
    return None, None

def monitor_vram():
    """Monitor VRAM in background."""
    global vram_data, stop_monitoring
    max_vram = 0

    while not stop_monitoring:
        used, total = get_vram()
        if used is not None:
            timestamp = datetime.now().isoformat()
            vram_data.append({"time": timestamp, "used_mb": used, "total_mb": total})

            if used > max_vram:
                max_vram = used
                print(f"[VRAM] New max: {used} MB / {total} MB ({100*used/total:.1f}%)")

            # Warning if close to limit
            if used > total * 0.9:
                print(f"[VRAM WARNING] {used} MB / {total} MB ({100*used/total:.1f}%) - CLOSE TO LIMIT!")

        time.sleep(MONITOR_INTERVAL)

    # Save log
    with open(VRAM_LOG, 'w') as f:
        json.dump(vram_data, f, indent=2)
    print(f"\n[VRAM] Log saved to {VRAM_LOG}")
    print(f"[VRAM] Max VRAM used: {max_vram} MB")

def main():
    global stop_monitoring

    print("=" * 60)
    print("SAM-MOSAIC WITH VRAM MONITORING")
    print("=" * 60)
    print(f"Input: {INPUT_IMAGE}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Monitor interval: {MONITOR_INTERVAL}s")
    print("=" * 60)

    # Start VRAM monitor
    monitor_thread = threading.Thread(target=monitor_vram, daemon=True)
    monitor_thread.start()

    # Run sam-mosaic
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["SAM_MOSAIC_DEBUG"] = "1"

    cmd = [
        "sam-mosaic",
        INPUT_IMAGE,
        OUTPUT_DIR,
        "--checkpoint", CHECKPOINT
    ]

    print(f"\nRunning: {' '.join(cmd)}\n")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in process.stdout:
            print(line, end='', flush=True)

        process.wait()

    except KeyboardInterrupt:
        print("\n[INTERRUPTED]")
        process.terminate()
    finally:
        stop_monitoring = True
        time.sleep(1)  # Let monitor save log

    print("\n" + "=" * 60)
    print("VRAM SUMMARY")
    print("=" * 60)

    if vram_data:
        vrams = [d["used_mb"] for d in vram_data]
        print(f"  Min:  {min(vrams)} MB")
        print(f"  Max:  {max(vrams)} MB")
        print(f"  Avg:  {sum(vrams)/len(vrams):.0f} MB")
        print(f"  Samples: {len(vrams)}")

        # Check for growth pattern
        if len(vrams) > 10:
            early = sum(vrams[:len(vrams)//5]) / (len(vrams)//5)
            late = sum(vrams[-len(vrams)//5:]) / (len(vrams)//5)
            growth = late - early
            print(f"  Early avg: {early:.0f} MB")
            print(f"  Late avg:  {late:.0f} MB")
            print(f"  Growth:    {growth:+.0f} MB")

            if growth > 1000:
                print("  [WARNING] Significant VRAM growth detected!")
            else:
                print("  [OK] VRAM stable")

if __name__ == "__main__":
    main()
