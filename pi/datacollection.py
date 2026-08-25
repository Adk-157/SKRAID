import os
import time
import csv
import threading
import subprocess
import signal
import datetime
import serial
import pynmea2
import queue

# ---------- CONFIG ----------
FPS = 30
VIDEO_DEVICE = "/dev/video0"

GPS_PORT = "/dev/serial0"
GPS_BAUD = 9600

IMU_PORT = "/dev/ttyUSB0"
IMU_BAUD = 57600

SAVE_DIR = "data_collection"
os.makedirs(SAVE_DIR, exist_ok=True)

SESSION_TS = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
CSV_PATH      = f"{SAVE_DIR}/fusion_{SESSION_TS}.csv"
VIDEO_PATH    = f"{SAVE_DIR}/video_{SESSION_TS}.mp4"
MANIFEST_PATH = f"{SAVE_DIR}/gps_frame_manifest_{SESSION_TS}.csv"  # NEW

# ---------- HEALTH MONITOR CONFIG ----------
GPS_TIMEOUT_SEC    = 5.0
IMU_TIMEOUT_SEC    = 2.0
CAM_TIMEOUT_SEC    = 3.0
HEALTH_CHECK_INTERVAL = 2.0

stop_event = threading.Event()
data_queue = queue.Queue()

sensor_last_seen = {
    "GPS": None,
    "IMU": None,
    "CAM": None,
}
sensor_lock = threading.Lock()

# ---------- NEW: shared frame_id counter ----------
# Camera timestamp thread increments this every frame.
# GPS thread reads it to log which frame was current at each GPS fix.
current_frame_id = 0
frame_id_lock    = threading.Lock()

def update_sensor_heartbeat(sensor_type):
    with sensor_lock:
        sensor_last_seen[sensor_type] = time.time()

# ---------- NEW: GPS-FRAME MANIFEST WRITER ----------
# A lightweight separate CSV that maps every GPS fix to the
# nearest camera frame_id and its wall-clock timestamp.
# Colab Cell 6 reads this to extract the exact frame from the
# AVI/MP4 without hunting through the full fusion CSV.
manifest_queue = queue.Queue()

def manifest_writer_thread():
    with open(MANIFEST_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "GPS_TIME",       # wall-clock timestamp of GPS fix
            "LAT",
            "LON",
            "SPEED_KMH",
            "FRAME_ID",       # camera frame_id at this GPS moment
            "FRAME_TIME_EST"  # estimated frame timestamp = frame_id / FPS seconds into session
        ])
        while not stop_event.is_set() or not manifest_queue.empty():
            try:
                row = manifest_queue.get(timeout=0.1)
                writer.writerow(row)
            except queue.Empty:
                pass

# ---------- CSV WRITER THREAD ----------
def csv_writer_thread():
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "TYPE", "TIME",
            "AX", "AY", "AZ",
            "ROLL", "PITCH", "YAW",
            "LAT", "LON", "SPEED_KMH",
            "FRAME_ID"
        ])
        while not stop_event.is_set() or not data_queue.empty():
            try:
                row = data_queue.get(timeout=0.1)
                writer.writerow(row)
            except queue.Empty:
                pass

# ---------- SENSOR HEALTH MONITOR ----------
def health_monitor_thread():
    timeouts = {
        "GPS": GPS_TIMEOUT_SEC,
        "IMU": IMU_TIMEOUT_SEC,
        "CAM": CAM_TIMEOUT_SEC,
    }
    print("[HEALTH] Sensor health monitor started. Waiting 10s for sensors to initialize...")
    time.sleep(10.0)

    while not stop_event.is_set():
        now = time.time()
        with sensor_lock:
            last_seen = dict(sensor_last_seen)

        all_ok = True
        for sensor, timeout in timeouts.items():
            last = last_seen[sensor]
            if last is None:
                print(f"[HEALTH] ⚠️  WARNING: {sensor} has never sent data since session start!")
                all_ok = False
            else:
                elapsed = now - last
                if elapsed > timeout:
                    print(f"[HEALTH] ⚠️  WARNING: {sensor} last seen {elapsed:.1f}s ago "
                          f"(timeout={timeout}s). Sensor may be disconnected!")

                    all_ok = False

        if all_ok:
            with sensor_lock:
                speeds = {k: (now - v) if v else None for k, v in sensor_last_seen.items()}
            print(f"[HEALTH] ✅ All sensors OK — "
                  f"GPS: {speeds['GPS']:.1f}s ago, "
                  f"IMU: {speeds['IMU']:.1f}s ago, "
                  f"CAM: {speeds['CAM']:.1f}s ago")

        time.sleep(HEALTH_CHECK_INTERVAL)

# ---------- CAMERA ----------
def start_camera():
    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-f", "v4l2",
            "-input_format", "mjpeg",
            "-video_size", "1280x720",
            "-framerate", str(FPS),
            "-i", VIDEO_DEVICE,
            "-vcodec", "libx264",   # encode directly to H264
            "-preset", "ultrafast", # lowest CPU load on Pi 4
            "-crf", "23",           # quality (18=best, 28=smallest, 23=balanced)
            "-pix_fmt", "yuv420p",  # ensures compatibility with cv2 in Colab
            VIDEO_PATH
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    print("Waiting for camera to initialize...")
    time.sleep(3.0)

    if proc.poll() is not None:
        raise RuntimeError(
            f"ffmpeg exited with code {proc.returncode}. "
            "Check VIDEO_DEVICE and that no other process holds the camera."
        )

    print("Camera ready.")
    return proc


def camera_timestamp_thread():
    global current_frame_id
    frame  = 0
    period = 1.0 / FPS

    while not stop_event.is_set():
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        data_queue.put(["CAM", now, "", "", "", "", "", "", "", "", "", frame])

        # Update shared frame counter so GPS thread can read it
        with frame_id_lock:
            current_frame_id = frame


        frame += 1
        update_sensor_heartbeat("CAM")
        time.sleep(period)

# ---------- GPS ----------
def gps_thread():
    try:
        ser = serial.Serial(GPS_PORT, GPS_BAUD, timeout=0.1)
    except Exception as e:
        print(f"[GPS] ❌ Failed to open port: {e}")
        return

    try:
        while not stop_event.is_set():
            try:
                line = ser.readline().decode(errors="ignore").strip()
            except serial.SerialException as e:
                print(f"[GPS] ❌ Serial read error: {e}")
                break

            if not line.startswith(("$GP", "$GN", "$BD")):
                continue

            try:
                msg = pynmea2.parse(line)
            except pynmea2.ParseError:
                continue

            if not (hasattr(msg, "latitude") and hasattr(msg, "longitude")):
                continue

            lat = msg.latitude  if msg.latitude  else 0.0
            lon = msg.longitude if msg.longitude else 0.0

            speed_kmh = 0.0
            if hasattr(msg, "spd_over_grnd") and msg.spd_over_grnd:
                try:
                    speed_kmh = float(msg.spd_over_grnd) * 1.852
                except (ValueError, TypeError):
                    speed_kmh = 0.0

            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

            # Snapshot the current frame_id at this exact GPS moment
            with frame_id_lock:
                snap_frame_id = current_frame_id

            # Estimate what timestamp this frame corresponds to in the video
            # frame_id / FPS = seconds elapsed since recording started
            frame_time_est = round(snap_frame_id / FPS, 3)

            # Write to main fusion CSV
            data_queue.put([
                "GPS", now,
                "", "", "",
                "", "", "",
                lat, lon, speed_kmh,
                snap_frame_id   # FRAME_ID column now filled for GPS rows
            ])

            # Write to manifest CSV for Colab frame extraction
            manifest_queue.put([
                now,
                round(lat, 7),
                round(lon, 7),
                round(speed_kmh, 3),
                snap_frame_id,
                frame_time_est  # e.g. 42.033 means seek to 42.033s in the video

            ])

            update_sensor_heartbeat("GPS")

    finally:
        ser.close()

# ---------- IMU ----------
def imu_thread():
    try:
        ser = serial.Serial(IMU_PORT, IMU_BAUD, timeout=0.01)
    except Exception as e:
        print(f"[IMU] ❌ Failed to open port: {e}")
        return

    buffer = ""
    try:
        while not stop_event.is_set():
            try:
                if ser.in_waiting > 0:
                    chunk  = ser.read(ser.in_waiting).decode(errors="ignore")
                    buffer += chunk

                    if '\n' in buffer or '\r' in buffer:
                        lines  = buffer.replace('\r', '\n').split('\n')
                        buffer = lines[-1]

                        for line in lines[:-1]:
                            line = line.strip()

                            if not line.startswith('#YPR='):
                                continue

                            parts = line.replace('#YPR=', '').split(',')
                            if len(parts) < 6:
                                continue

                            try:
                                yaw   = float(parts[0])
                                pitch = float(parts[1])
                                roll  = float(parts[2])
                                acc_x = float(parts[3].replace('ACC=', ''))
                                acc_y = float(parts[4])
                                acc_z = float(parts[5])
                            except ValueError:
                                continue

                            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                            data_queue.put([
                                "IMU", now,
                                acc_x, acc_y, acc_z,
                                roll, pitch, yaw,
                                "", "", "",
                                ""
                            ])
                            update_sensor_heartbeat("IMU")
                else:
                    time.sleep(0.005)

            except serial.SerialException as e:
                print(f"[IMU] ❌ Serial error: {e}")
                break

    finally:
        ser.close()

# ---------- MAIN ----------
def main():
    print(f"Starting capture... Saving to {SAVE_DIR}")
    print(f"  Video    : {VIDEO_PATH}")
    print(f"  CSV      : {CSV_PATH}")
    print(f"  Manifest : {MANIFEST_PATH}")
    print("-" * 50)

    try:
        cam_proc = start_camera()
    except RuntimeError as e:
        print(f"Camera startup failed: {e}")
        return

    writer_t   = threading.Thread(target=csv_writer_thread,      daemon=False)
    manifest_t = threading.Thread(target=manifest_writer_thread, daemon=False)  # NEW
    cam_ts_t   = threading.Thread(target=camera_timestamp_thread, daemon=True)
    gps_t      = threading.Thread(target=gps_thread,             daemon=True)
    imu_t      = threading.Thread(target=imu_thread,             daemon=True)
    health_t   = threading.Thread(target=health_monitor_thread,  daemon=True)

    for t in [writer_t, manifest_t, cam_ts_t, gps_t, imu_t, health_t]:
        t.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        stop_event.set()

    cam_proc.send_signal(signal.SIGINT)
    try:
        cam_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        print("ffmpeg did not stop cleanly — killing.")
        cam_proc.kill()

    writer_t.join(timeout=5.0)
    manifest_t.join(timeout=5.0)   # NEW

    print(f"\nSession complete.")
    print(f"  Video    : {VIDEO_PATH}")
    print(f"  CSV      : {CSV_PATH}")
    print(f"  Manifest : {MANIFEST_PATH}")

if __name__ == "__main__":
    main()