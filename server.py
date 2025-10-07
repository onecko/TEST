import os
import sys
import socket
import threading
import time
import shutil
import csv
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import paramiko
from paramiko import SFTPServerInterface, SFTPHandle, SFTPAttributes, SFTPServer
from paramiko.sftp import SFTP_OK, SFTP_FAILURE

import cv2
import numpy as np
from batch_face import RetinaFace, SixDRep


HOST = "0.0.0.0"
PORT = 5022
USERNAME = "ser_user"
PASSWORD = "ur;\\8:3D0po8ym]"
BASE_DIR = Path("/home/ser_user/server")
INCOMING_DIR = BASE_DIR / "data" / "incoming"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REJECTED_DIR = BASE_DIR / "data" / "rejected"
OUTPUT_DIR = BASE_DIR / "data" / "output"
LOGS_DIR = BASE_DIR / "logs"
HOST_KEY_FILE = BASE_DIR / "host_key.key"

for dir_path in [INCOMING_DIR, PROCESSED_DIR, REJECTED_DIR, OUTPUT_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True, mode=0o755)

print(f"Starting Enhanced SFTP Server with Head Pose Detection")
print(f"Server: {socket.gethostname()} ({socket.gethostbyname(socket.gethostname())})")
print(f"Listening on: {HOST}:{PORT}")
print(f"Base Directory: {BASE_DIR}")
print(f"Directories:")
print(f"  Incoming:  {INCOMING_DIR}")
print(f"  Processed: {PROCESSED_DIR}")
print(f"  Rejected:  {REJECTED_DIR}")
print(f"  Output:    {OUTPUT_DIR}")
print(f"  Logs:      {LOGS_DIR}")

stats = {
    'received': 0,
    'processed': 0,
    'rejected': 0,
    'connections': 0,
    'faces_detected': 0,
    'no_faces': 0
}

stats_lock = threading.Lock()
direction_stats = defaultdict(lambda: {"left": 0, "center": 0, "right": 0, "up": 0, "down": 0})

if HOST_KEY_FILE.exists():
    try:
        host_key = paramiko.RSAKey(filename=str(HOST_KEY_FILE))
        print("Loaded existing host key")
    except Exception as e:
        print(f"Error loading host key: {e}")
        host_key = paramiko.RSAKey.generate(2048)
        host_key.write_private_key_file(str(HOST_KEY_FILE))
        print("Generated new host key (old key was invalid)")
else:
    host_key = paramiko.RSAKey.generate(2048)
    host_key.write_private_key_file(str(HOST_KEY_FILE))
    os.chmod(str(HOST_KEY_FILE), 0o600)  # Set proper permissions for security
    print("Generated new host key")


class HeadPoseProcessor:
    def __init__(self):
        print("[AI] Initializing head pose detection models...")
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            gpu_id = 0 if cuda_available else -1

            if not cuda_available:
                print("[AI] CUDA not available, using CPU mode")
            else:
                print(f"[AI] CUDA available, using GPU {gpu_id}")

            self.detector = RetinaFace(gpu_id=gpu_id)
            self.estimator = SixDRep(gpu_id=gpu_id)
            print("[AI] Models initialized successfully")
            self.initialized = True
        except Exception as e:
            print(f"[AI ERROR] Failed to initialize models: {e}")
            print("[AI] Running in fallback mode (no AI processing)")
            self.initialized = False

    def get_direction(self, yaw, pitch):
        horizontal = "center"
        vertical = "center"

        if yaw < -20:
            horizontal = "left"
        elif yaw > 20:
            horizontal = "right"

        if pitch < -15:
            vertical = "down"
        elif pitch > 15:
            vertical = "up"

        return horizontal, vertical

    def draw_axes(self, img, yaw, pitch, roll, tdx, tdy, size=80):
        yaw, pitch, roll = map(np.radians, [yaw, pitch, roll])

        Rx = np.array([[1, 0, 0],
                       [0, np.cos(pitch), -np.sin(pitch)],
                       [0, np.sin(pitch), np.cos(pitch)]])
        Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                       [0, 1, 0],
                       [-np.sin(yaw), 0, np.cos(yaw)]])
        Rz = np.array([[np.cos(roll), -np.sin(roll), 0],
                       [np.sin(roll), np.cos(roll), 0],
                       [0, 0, 1]])

        R = Rz @ Ry @ Rx
        axis = np.eye(3) * size
        pts3d = R @ axis

        xaxis = (int(tdx + pts3d[0, 0]), int(tdy - pts3d[1, 0]))
        yaxis = (int(tdx + pts3d[0, 1]), int(tdy - pts3d[1, 1]))
        zaxis = (int(tdx + pts3d[0, 2]), int(tdy - pts3d[1, 2]))

        cv2.arrowedLine(img, (tdx, tdy), xaxis, (0, 0, 255), 2)
        cv2.arrowedLine(img, (tdx, tdy), yaxis, (0, 255, 0), 2)
        cv2.arrowedLine(img, (tdx, tdy), zaxis, (255, 0, 0), 2)

        return img

    def process_image(self, image_path):
        if not self.initialized:
            return None

        try:
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"[AI ERROR] Failed to read image: {image_path}")
                return None

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            faces = self.detector(rgb, threshold=0.5, return_dict=True)

            if not faces:
                print(f"[AI] No faces detected in {image_path.name}")
                stats['no_faces'] += 1
                return None

            all_faces = [faces]
            frames = [rgb]
            self.estimator(all_faces, frames, update_dict=True, input_face_type='dict')

            faces0 = all_faces[0]
            timestamp = datetime.now()
            results = []

            for face_idx, face in enumerate(faces0):
                box = face['box']
                hp = face.get('head_pose')

                if hp is None:
                    continue

                yaw, pitch, roll = hp['yaw'], hp['pitch'], hp['roll']
                horizontal, vertical = self.get_direction(yaw, pitch)

                log_entry = {
                    'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'image': image_path.name,
                    'face_id': face_idx + 1,
                    'yaw': round(yaw, 2),
                    'pitch': round(pitch, 2),
                    'roll': round(roll, 2),
                    'horizontal': horizontal,
                    'vertical': vertical
                }

                log_file = LOGS_DIR / f"head_pose_{timestamp.strftime('%Y%m%d')}.log"
                with open(log_file, "a") as f:
                    f.write(f"{log_entry['timestamp']} | {log_entry['image']} | "
                            f"Face {log_entry['face_id']} | "
                            f"Yaw={log_entry['yaw']}° Pitch={log_entry['pitch']}° Roll={log_entry['roll']}° | "
                            f"Looking: {log_entry['horizontal']}, {log_entry['vertical']}\n")

                with stats_lock:
                    time_slot = timestamp.strftime("%Y-%m-%d %H:%M")
                    if horizontal != "center":
                        direction_stats[time_slot][horizontal] += 1
                    if vertical != "center":
                        direction_stats[time_slot][vertical] += 1
                    if horizontal == "center" and vertical == "center":
                        direction_stats[time_slot]["center"] += 1

                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                self.draw_axes(img, yaw, pitch, roll, cx, cy, size=max(x2 - x1, y2 - y1) // 2)

                cv2.putText(img, f"Face {face_idx + 1}: {horizontal}, {vertical}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(img, f"Y:{yaw:.1f} P:{pitch:.1f} R:{roll:.1f}",
                            (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

                results.append(log_entry)
                stats['faces_detected'] += 1

            output_path = OUTPUT_DIR / f"annotated_{timestamp.strftime('%Y%m%d_%H%M%S')}_{image_path.name}"
            cv2.imwrite(str(output_path), img)
            print(f"[AI] Saved annotated image: {output_path.name}")

            return results

        except Exception as e:
            print(f"[AI ERROR] Processing failed for {image_path}: {e}")
            return None


ai_processor = None


class MySFTPHandle(SFTPHandle):
    def __init__(self, flags=0):
        super().__init__(flags)
        self.file = None

    def close(self):
        if self.file:
            self.file.close()
        return SFTP_OK

    def read(self, offset, length):
        if self.file:
            self.file.seek(offset)
            return self.file.read(length)
        return SFTP_FAILURE

    def write(self, offset, data):
        if self.file:
            self.file.seek(offset)
            self.file.write(data)
            self.file.flush()
            return SFTP_OK
        return SFTP_FAILURE

    def stat(self):
        if self.file:
            return SFTPAttributes.from_stat(os.fstat(self.file.fileno()))
        return SFTP_FAILURE


class MySFTPServer(SFTPServerInterface):
    def __init__(self, server, *args, **kwargs):
        super().__init__(server, *args, **kwargs)
        self.ROOT = str(INCOMING_DIR)

    def _realpath(self, path):
        if path == "/" or path == ".":
            return self.ROOT
        path = path.replace("\\", "/")
        if path.startswith("/"):
            path = path[1:]
        realpath = os.path.join(self.ROOT, path)
        realpath = os.path.normpath(realpath)
        if not realpath.startswith(self.ROOT):
            return self.ROOT
        return realpath

    def list_folder(self, path):
        realpath = self._realpath(path)
        try:
            out = []
            for fname in os.listdir(realpath):
                full_path = os.path.join(realpath, fname)
                attr = SFTPAttributes.from_stat(os.stat(full_path))
                attr.filename = fname
                out.append(attr)
            return out
        except Exception as e:
            print(f"[SFTP] Error listing folder {path}: {e}")
            return []

    def open(self, path, flags, attr):
        global stats
        realpath = self._realpath(path)
        filename = os.path.basename(realpath)

        try:
            handle = MySFTPHandle(flags)

            if flags & os.O_WRONLY:
                mode = 'wb'
            elif flags & os.O_RDWR:
                mode = 'r+b'
            else:
                mode = 'rb'

            handle.file = open(realpath, mode)

            if flags & os.O_WRONLY:
                stats['received'] += 1
                print(f"[RECEIVED] {filename} - Total: {stats['received']}")

            return handle

        except Exception as e:
            try:
                handle = MySFTPHandle(flags)
                handle.file = open(realpath, 'wb')
                stats['received'] += 1
                print(f"[RECEIVED] {filename} - Total: {stats['received']}")
                return handle
            except:
                return SFTP_FAILURE

    def remove(self, path):
        realpath = self._realpath(path)
        try:
            os.remove(realpath)
            return SFTP_OK
        except:
            return SFTP_FAILURE

    def rename(self, oldpath, newpath):
        old_realpath = self._realpath(oldpath)
        new_realpath = self._realpath(newpath)
        try:
            os.rename(old_realpath, new_realpath)
            return SFTP_OK
        except:
            return SFTP_FAILURE

    def mkdir(self, path, attr):
        realpath = self._realpath(path)
        try:
            os.makedirs(realpath, exist_ok=True)
            return SFTP_OK
        except:
            return SFTP_FAILURE

    def rmdir(self, path):
        realpath = self._realpath(path)
        try:
            os.rmdir(realpath)
            return SFTP_OK
        except:
            return SFTP_FAILURE

    def stat(self, path):
        realpath = self._realpath(path)
        try:
            return SFTPAttributes.from_stat(os.stat(realpath))
        except:
            return SFTP_FAILURE

    def lstat(self, path):
        return self.stat(path)


# SSH Server
class MySSHServer(paramiko.ServerInterface):
    def check_auth_password(self, username, password):
        if username == USERNAME and password == PASSWORD:
            return paramiko.AUTH_SUCCESSFUL
        return paramiko.AUTH_FAILED

    def check_channel_request(self, kind, chanid):
        if kind == 'session':
            return paramiko.OPEN_SUCCEEDED
        return paramiko.OPEN_FAILED_ADMINISTRATIVELY_PROHIBITED

    def get_allowed_auths(self, username):
        return "password"


# File Processor Thread with AI
def file_processor_with_ai():
    """Process files from incoming to processed directory with head pose detection"""
    global stats, ai_processor
    print("[PROCESSOR] Started with AI capabilities")

    while True:
        try:
            # Get all image files in incoming directory
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

            for file_path in INCOMING_DIR.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                    # Wait a bit to ensure file is fully written
                    time.sleep(0.5)

                    # Check file size (basic validation)
                    file_size = file_path.stat().st_size

                    if file_size < 1000:  # Less than 1KB, probably corrupted
                        # Move to rejected
                        dest = REJECTED_DIR / file_path.name
                        shutil.move(str(file_path), str(dest))
                        stats['rejected'] += 1
                        print(f"[REJECTED] {file_path.name} (too small: {file_size} bytes)")
                    else:
                        # Process with AI if available
                        ai_results = None
                        if ai_processor and ai_processor.initialized:
                            print(f"[AI PROCESSING] {file_path.name}")
                            ai_results = ai_processor.process_image(file_path)

                        # Move to processed
                        dest = PROCESSED_DIR / file_path.name

                        if dest.exists():
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            dest = PROCESSED_DIR / f"{file_path.stem}_{timestamp}{file_path.suffix}"

                        shutil.move(str(file_path), str(dest))
                        stats['processed'] += 1

                        # Log results
                        if ai_results:
                            print(f"[AI RESULTS] {file_path.name}: {len(ai_results)} faces detected")
                            for result in ai_results:
                                print(f"  Face {result['face_id']}: {result['horizontal']}, {result['vertical']}")

                        success_rate = (stats['processed'] / stats['received'] * 100) if stats['received'] > 0 else 0
                        face_detection_rate = (stats['faces_detected'] / stats['processed'] * 100) if stats[
                                                                                                          'processed'] > 0 else 0

                        print(f"[PROCESSED] {file_path.name} ({file_size:,} bytes)")
                        print(f"[STATS] Received: {stats['received']}, "
                              f"Processed: {stats['processed']}, "
                              f"Rejected: {stats['rejected']}, "
                              f"Faces: {stats['faces_detected']}, "
                              f"Success: {success_rate:.1f}%, "
                              f"Face Rate: {face_detection_rate:.1f}%")

            time.sleep(2)

        except Exception as e:
            print(f"[PROCESSOR ERROR] {e}")
            time.sleep(5)


def write_statistics():
    while True:
        time.sleep(60)

        with stats_lock:
            if not direction_stats:
                continue

            timestamp = datetime.now()
            csv_file = LOGS_DIR / f"statistics_{timestamp.strftime('%Y%m%d')}.csv"

            period_stats = {"left": 0, "right": 0, "up": 0, "down": 0, "center": 0}
            for time_slot, counts in direction_stats.items():
                for direction, count in counts.items():
                    period_stats[direction] += count

            file_exists = csv_file.exists()
            with open(csv_file, 'a', newline='') as f:
                fieldnames = ['timestamp', 'left', 'right', 'up', 'down', 'center', 'total']
                writer = csv.DictWriter(f, fieldnames=fieldnames)

                if not file_exists:
                    writer.writeheader()

                total = sum(period_stats.values())
                writer.writerow({
                    'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'left': period_stats['left'],
                    'right': period_stats['right'],
                    'up': period_stats['up'],
                    'down': period_stats['down'],
                    'center': period_stats['center'],
                    'total': total
                })

            print(f"[STATISTICS] Updated: {total} detections logged")
            direction_stats.clear()


def handle_client(client_socket, address):
    global stats
    stats['connections'] += 1

    print(f"\n[CONNECTION #{stats['connections']}] {address[0]}:{address[1]}")

    transport = None
    try:
        transport = paramiko.Transport(client_socket)
        transport.add_server_key(host_key)

        transport.set_subsystem_handler(
            'sftp',
            paramiko.SFTPServer,
            sftp_si=MySFTPServer
        )

        server = MySSHServer()
        transport.start_server(server=server)

        channel = transport.accept(20)
        if channel is None:
            print(f"[FAILED] No channel from {address[0]}")
            return

        print(f"[ACTIVE] Session from {address[0]}")

        while transport.is_active():
            threading.Event().wait(0.5)

    except Exception as e:
        print(f"[ERROR] {address[0]}: {e}")
    finally:
        if transport:
            transport.close()
        print(f"[CLOSED] {address[0]}:{address[1]}")


def write_pid_file():
    pid_file = BASE_DIR / 'sftp_server.pid'
    with open(pid_file, 'w') as f:
        f.write(str(os.getpid()))
    print(f"PID {os.getpid()} written to {pid_file}")


def remove_pid_file():
    pid_file = BASE_DIR / 'sftp_server.pid'
    if pid_file.exists():
        pid_file.unlink()
        print(f"PID file removed")


def signal_handler(signum, frame):
    print(f"\nReceived signal {signum}, shutting down gracefully...")
    remove_pid_file()

    print("\n" + "=" * 60)
    print("FINAL STATISTICS:")
    print(f"  Connections:    {stats['connections']}")
    print(f"  Received:       {stats['received']}")
    print(f"  Processed:      {stats['processed']}")
    print(f"  Rejected:       {stats['rejected']}")
    print(f"  Faces Detected: {stats['faces_detected']}")
    print(f"  No Faces:       {stats['no_faces']}")

    if stats['received'] > 0:
        success_rate = (stats['processed'] / stats['received'] * 100)
        print(f"  Success Rate:   {success_rate:.1f}%")

    if stats['processed'] > 0:
        face_rate = (stats['faces_detected'] / stats['processed'] * 100)
        print(f"  Face Det. Rate: {face_rate:.1f}%")

    final_stats_file = LOGS_DIR / f"final_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(final_stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nFinal statistics saved to: {final_stats_file}")
    print("=" * 60)

    sys.exit(0)


def main():
    global ai_processor

    import signal
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    write_pid_file()

    print("\n" + "=" * 40)
    print("INITIALIZING AI COMPONENTS")
    print("=" * 40)
    ai_processor = HeadPoseProcessor()

    processor_thread = threading.Thread(target=file_processor_with_ai, daemon=True)
    processor_thread.start()

    stats_thread = threading.Thread(target=write_statistics, daemon=True)
    stats_thread.start()

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        server_socket.bind((HOST, PORT))
    except OSError as e:
        print(f"Cannot bind to {HOST}:{PORT}")
        print(f"Error: {e}")
        print("Possible causes:")
        print("  1. Port is already in use")
        print("  2. Need to run with sudo for ports < 1024")
        print("  3. Check if another instance is running")
        remove_pid_file()
        sys.exit(1)

    server_socket.listen(5)

    print("\n" + "=" * 40)
    print(f"SERVER READY WITH HEAD POSE DETECTION")
    print("=" * 40)
    print(f"Server IP: 192.168.0.228")
    print(f"Listening: {HOST}:{PORT}")
    print(f"Username: {USERNAME}")
    print(f"Password: [configured]")
    print(f"Base Dir: {BASE_DIR}")
    print("\nWaiting for connections...")
    print("Press Ctrl+C to stop")
    print("=" * 60)

    try:
        while True:
            client_socket, address = server_socket.accept()
            thread = threading.Thread(
                target=handle_client,
                args=(client_socket, address),
                daemon=True
            )
            thread.start()

    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)
    finally:
        server_socket.close()
        remove_pid_file()


if __name__ == "__main__":
    import pwd
    current_user = pwd.getpwuid(os.getuid()).pw_name
    if current_user != 'ser_user':
        print(f"Warning: Running as '{current_user}' instead of 'ser_user'")
        print("Some permissions might be incorrect.")

    main()