import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import threading
import os
import time
import urllib.request
import json

from pipeline.gear_core      import run_core_pipeline
from pipeline.gear_mask      import build_gear_mask
from pipeline.tooth_analysis import (get_contour_signal,
                                      detect_teeth_from_contour,
                                      measure_teeth_from_contour)
from pipeline.sideProfile    import run_sideprofile_pipeline

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════
DEFAULT_TOLERANCES = {
    'tip_diameter_mm'   : (39.0,  41.0),
    'root_diameter_mm'  : (37.5,  39.0),
    'pitch_diameter_mm' : (38.5,  40.0),
    'shaft_diameter_mm' : (2.8,   3.3),
    'tooth_count'       : (71,    71),
    'tooth_width_mm'    : (1.4,   2.0),
    'tooth_depth_mm'    : (0.7,   1.2),
    'circularity'       : (0.75,  1.0),
    'face_width_mm'     : (4.0,   5.5),
    'perp_deviation_deg': (0.0,   2.0),
    'helix_angle_deg'   : (18.0,  24.0),
}

PX_PER_MM      = 20.0
EXPECTED_TEETH = None
GEAR_TYPE      = "helical"

# Temporary capture paths
CAPTURE_ENDFACE   = "results/_capture_endface.jpg"
CAPTURE_SIDE      = "results/_capture_side.jpg"

CONFIG_FILE = "gear_inspection_config.json"

# ── Overlay colours ────────────────────────────────────────────────────────
OVL_CIRCLE  = (0,   200, 255)   # orange-yellow  ring guide
OVL_CENTRE  = (0,   255, 0  )   # green          centre dot / crosshair
OVL_LINES   = (0,   200, 255)   # orange-yellow  horizontal guides
OVL_READY   = (0,   255, 0  )   # green          "gear in zone" flash
OVL_ALPHA   = 0.35              # overlay transparency


# ══════════════════════════════════════════════════════════════════════════
# CAMERA STREAM  (IP Webcam over WiFi)
# ══════════════════════════════════════════════════════════════════════════
class IPCamera:
    """
    Reads frames from an IP Webcam (Android) MJPEG stream.

    Stream URL format:  http://<phone_ip>:8080/video
    Snapshot URL:       http://<phone_ip>:8080/shot.jpg

    Setup on phone:
      1. Install "IP Webcam" by Pavel Khlebovich (Play Store — free)
      2. Open app → scroll to bottom → tap "Start server"
      3. Note the IP address shown on screen
      4. Enter that IP in the dashboard
    """

    def __init__(self, url=""):
        self.url        = url
        self.cap        = None
        self.connected  = False
        self.last_frame = None
        self._lock      = threading.Lock()
        self._thread    = None
        self._running   = False

    def connect(self, url):
        """Connect to IP Webcam stream."""
        self.disconnect()
        self.url = url.strip()
        if not self.url:
            return False

        if not self.url.startswith("http"):
            self.url = f"http://{self.url}:8080"
        stream_url = self.url.rstrip("/") + "/video"

        try:
            self.cap = cv2.VideoCapture(stream_url)

            # ── Key settings for low latency ──────────────────────────────
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, 15)
            # ──────────────────────────────────────────────────────────────

            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.cap.release()
                self.connected = False
                return False

            with self._lock:
                self.last_frame = frame
            self.connected = True
            self._running  = True
            self._thread   = threading.Thread(
                target=self._read_loop, daemon=True)
            self._thread.start()
            return True

        except Exception:
            self.connected = False
            return False
    
    def _read_loop(self):
        """Background thread — reads frames, always keeps latest only."""
        while self._running and self.cap and self.cap.isOpened():
            # Grab without decoding to flush buffer
            grabbed = self.cap.grab()
            if not grabbed:
                time.sleep(0.02)
                continue

            # Only decode every Nth grab to keep CPU low
            # This ensures we always show the most recent frame
            ret, frame = self.cap.retrieve()
            if ret and frame is not None:
                with self._lock:
                    self.last_frame = frame
    
    def get_frame(self):
        """Returns the latest frame (BGR) or None."""
        with self._lock:
            return (self.last_frame.copy()
                    if self.last_frame is not None else None)

    def capture_snapshot(self):
        """
        Captures a high-quality snapshot via the /shot.jpg endpoint.
        Falls back to current video frame if snapshot fails.
        """
        snap_url = self.url.rstrip("/") + "/shot.jpg"
        try:
            resp  = urllib.request.urlopen(snap_url, timeout=3)
            data  = np.frombuffer(resp.read(), dtype=np.uint8)
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if frame is not None:
                return frame
        except Exception:
            pass
        return self.get_frame()

    def disconnect(self):
        self._running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.connected  = False
        self.last_frame = None


# ══════════════════════════════════════════════════════════════════════════
# OVERLAY DRAWING
# ══════════════════════════════════════════════════════════════════════════
def draw_endface_overlay(frame, gear_in_zone=False):
    """
    Draws placement guide on endface (top-down) camera feed.

    Overlay elements:
      - Large circle  : where gear outer edge should sit
      - Small circle  : where shaft hole should appear
      - Crosshairs    : gear centre
      - Corner marks  : frame boundary reference
      - Status text   : positioning instruction
    """
    h, w  = frame.shape[:2]
    cx, cy = w // 2, h // 2

    overlay = frame.copy()

    # Outer gear guide circle — 70% of shorter dimension
    gear_r  = int(min(w, h) * 0.35)
    # Shaft guide circle — ~8% of gear radius
    shaft_r = int(gear_r * 0.08)

    colour = OVL_READY if gear_in_zone else OVL_CIRCLE

    # Outer gear ring
    cv2.circle(overlay, (cx, cy), gear_r,  colour, 2)
    # Inner dashed ring (tooth root approximation)
    root_r = int(gear_r * 0.95)
    cv2.circle(overlay, (cx, cy), root_r, colour, 1)
    # Shaft hole guide
    cv2.circle(overlay, (cx, cy), shaft_r, OVL_CENTRE, 2)
    # Centre dot
    cv2.circle(overlay, (cx, cy), 4, OVL_CENTRE, -1)

    # Crosshairs
    cv2.line(overlay, (cx - gear_r, cy), (cx + gear_r, cy),
             OVL_CENTRE, 1)
    cv2.line(overlay, (cx, cy - gear_r), (cx, cy + gear_r),
             OVL_CENTRE, 1)

    # Corner bracket marks
    blen = 30
    bthk = 2
    corners = [(20, 20), (w-20, 20), (20, h-20), (w-20, h-20)]
    for bx, by in corners:
        dx = 1 if bx < w//2 else -1
        dy = 1 if by < h//2 else -1
        cv2.line(overlay, (bx, by), (bx + dx*blen, by), colour, bthk)
        cv2.line(overlay, (bx, by), (bx, by + dy*blen), colour, bthk)

    # Blend overlay
    out = cv2.addWeighted(overlay, OVL_ALPHA,
                          frame,   1 - OVL_ALPHA, 0)

    # Status text
    if gear_in_zone:
        txt = "GEAR IN POSITION  —  Ready to capture"
        tc  = (0, 255, 0)
    else:
        txt = "Align gear within circle  |  Shaft hole at centre"
        tc  = (255, 255, 255)

    cv2.putText(out, txt, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
    cv2.putText(out, txt, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, tc, 1)

    # Label
    cv2.putText(out, "TOP-DOWN VIEW", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0,0,0), 3)
    cv2.putText(out, "TOP-DOWN VIEW", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                colour, 1)
    return out


def draw_sideprofile_overlay(frame, gear_in_zone=False):
    """
    Draws placement guide on side profile camera feed.

    Overlay elements:
      - Two horizontal lines : where top/bottom gear face edges should sit
      - Vertical centre line : gear should be centred horizontally
      - Zone rectangle       : target region for gear face
      - Status text
    """
    h, w  = frame.shape[:2]
    cx    = w // 2

    overlay = frame.copy()
    colour  = OVL_READY if gear_in_zone else OVL_LINES

    # Horizontal guide lines — gear face should sit between these
    # Target: face occupies middle 20% of frame height
    top_y = int(h * 0.38)
    bot_y = int(h * 0.62)

    # Zone rectangle (semi-transparent fill)
    cv2.rectangle(overlay, (int(w*0.1), top_y),
                  (int(w*0.9), bot_y), colour, -1)

    blend = cv2.addWeighted(overlay, 0.12, frame, 0.88, 0)

    # Top and bottom edge lines
    cv2.line(blend, (0, top_y), (w, top_y), colour, 2)
    cv2.line(blend, (0, bot_y), (w, bot_y), colour, 2)

    # Vertical centre line
    cv2.line(blend, (cx, 0), (cx, h), OVL_CENTRE, 1)

    # Tick marks along horizontal lines
    for x in range(0, w, 40):
        cv2.line(blend, (x, top_y-6), (x, top_y+6), colour, 1)
        cv2.line(blend, (x, bot_y-6), (x, bot_y+6), colour, 1)

    # Face width arrow indicator
    arr_x = int(w * 0.07)
    cv2.arrowedLine(blend, (arr_x, top_y), (arr_x, bot_y),
                    colour, 1, tipLength=0.1)
    cv2.arrowedLine(blend, (arr_x, bot_y), (arr_x, top_y),
                    colour, 1, tipLength=0.1)
    cv2.putText(blend, "Face", (arr_x+4, (top_y+bot_y)//2 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1)
    cv2.putText(blend, "Width", (arr_x+4, (top_y+bot_y)//2 + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1)

    # Corner brackets
    blen = 30
    corners = [(int(w*0.1), top_y), (int(w*0.9), top_y),
               (int(w*0.1), bot_y), (int(w*0.9), bot_y)]
    for bx, by in corners:
        dx = 1 if bx < w//2 else -1
        dy = 1 if by < h//2 else -1
        cv2.line(blend, (bx, by), (bx+dx*blen, by), colour, 2)
        cv2.line(blend, (bx, by), (bx, by+dy*blen), colour, 2)

    # Status text
    if gear_in_zone:
        txt = "GEAR IN POSITION  —  Ready to capture"
        tc  = (0, 255, 0)
    else:
        txt = "Align gear face between the lines  |  Centre horizontally"
        tc  = (255, 255, 255)

    cv2.putText(blend, txt, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
    cv2.putText(blend, txt, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, tc, 1)

    # Label
    cv2.putText(blend, "SIDE PROFILE VIEW", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3)
    cv2.putText(blend, "SIDE PROFILE VIEW", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1)

    return blend


def check_endface_position(frame):
    """
    Checks if the gear is roughly in the correct position
    for the endface view.

    Looks for a large circular object centred in the frame.
    Returns True if gear appears centred and fills the guide circle.
    """
    try:
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w    = gray.shape
        cx, cy  = w // 2, h // 2
        gear_r  = int(min(w, h) * 0.35)

        # Sample brightness on the guide circle
        angles  = np.linspace(0, 2*np.pi, 72, endpoint=False)
        xs = np.clip((cx + gear_r * np.cos(angles)).astype(int), 0, w-1)
        ys = np.clip((cy + gear_r * np.sin(angles)).astype(int), 0, h-1)
        ring_mean = gray[ys, xs].mean()

        # Sample brightness inside (gear body)
        inner_r = int(gear_r * 0.6)
        xs_in   = np.clip(
            (cx + inner_r * np.cos(angles)).astype(int), 0, w-1)
        ys_in   = np.clip(
            (cy + inner_r * np.sin(angles)).astype(int), 0, h-1)
        inner_mean = gray[ys_in, xs_in].mean()

        # Gear is likely in position if guide ring hits the dark
        # boundary between gear and background
        contrast = abs(float(ring_mean) - float(inner_mean))
        return contrast > 15 and inner_mean > 30

    except Exception:
        return False


def check_sideprofile_position(frame):
    """
    Checks if gear face is visible in the guide zone.
    Returns True if horizontal edges are detected near guide lines.
    """
    try:
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w    = gray.shape
        top_y   = int(h * 0.38)
        bot_y   = int(h * 0.62)

        blurred = cv2.GaussianBlur(gray, (5,5), sigmaX=0)
        edges   = cv2.Canny(blurred, 30, 90)

        # Check for horizontal edges near guide lines (±20px)
        top_zone = edges[max(0,top_y-20):top_y+20, :]
        bot_zone = edges[max(0,bot_y-20):bot_y+20, :]

        top_edges = np.sum(top_zone > 0)
        bot_edges = np.sum(bot_zone > 0)

        return top_edges > w * 0.15 and bot_edges > w * 0.15

    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════
# PIPELINE RUNNERS
# ══════════════════════════════════════════════════════════════════════════
def run_endface_pipeline(image_path):
    m, inter = run_core_pipeline(image_path, px_per_mm=PX_PER_MM)
    gray     = inter['gray']
    gcx      = m['gear_centre_x']
    gcy      = m['gear_centre_y']
    tip_r    = inter['tip_radius_px']
    root_r   = inter['root_radius_px']
    shaft_r  = inter['shaft_radius_px']

    gear_only, _ = build_gear_mask(
        gray, gcx, gcy, root_r * 0.15, tip_r)
    angles, dists, pts = get_contour_signal(
        gear_only, gcx, gcy, tip_radius_px=tip_r)
    tooth_peaks, gap_valleys, smoothed = detect_teeth_from_contour(
        dists, angles, expected_count=EXPECTED_TEETH)
    _, summary = measure_teeth_from_contour(
        dists, angles, pts,
        tooth_peaks, gap_valleys, smoothed,
        gcx, gcy, PX_PER_MM)

    tip_r_fit  = summary.get('tip_radius_fitted_px',  tip_r)
    root_r_fit = summary.get('root_radius_fitted_px', root_r)
    pitch_r    = tip_r_fit - (tip_r_fit - root_r_fit) / 2.25

    meas = {
        'tip_diameter_mm'    : tip_r_fit  * 2 / PX_PER_MM,
        'root_diameter_mm'   : root_r_fit * 2 / PX_PER_MM,
        'pitch_diameter_mm'  : pitch_r    * 2 / PX_PER_MM,
        'shaft_diameter_mm'  : m['shaft_diameter_px'] / PX_PER_MM,
        'tooth_count'        : summary.get('tooth_count', 0),
        'tooth_width_mm'     : summary.get('tooth_width_mean_mm', 0),
        'tooth_depth_mm'     : summary.get('tooth_depth_mean_mm', 0),
        'circularity'        : m['circularity'],
        'circularity_gdt_mm' : m.get('circularity_gdt_zone', 0) / PX_PER_MM,
        'gear_cx'            : gcx,
        'gear_cy'            : gcy,
        '_tip_r'             : tip_r_fit,
        '_root_r'            : root_r_fit,
        '_pitch_r'           : pitch_r,
        '_shaft_r'           : shaft_r,
        '_shaft_ellipse'     : inter.get('shaft_ellipse'),
    }

    # Annotated image
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.circle(vis, (gcx, gcy), int(tip_r_fit),  (0, 140, 255), 2)
    cv2.circle(vis, (gcx, gcy), int(root_r_fit), (0,   0, 255), 2)
    cv2.circle(vis, (gcx, gcy), int(pitch_r),    (0, 255,   0), 2)
    cv2.circle(vis, (gcx, gcy), int(shaft_r),    (255, 255, 0), 2)
    cv2.line(vis, (gcx-40,gcy), (gcx+40,gcy), (255,255,255), 1)
    cv2.line(vis, (gcx,gcy-40), (gcx,gcy+40), (255,255,255), 1)
    return meas, vis


def run_side_pipeline(image_path):
    m, debug = run_sideprofile_pipeline(
        image_path, GEAR_TYPE, PX_PER_MM)

    gray  = debug['gray']
    face  = debug['face']
    shaft = debug['shaft']
    helix = debug['helix']
    h, w  = gray.shape

    vis  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    x_arr   = np.array([face['x_start'], face['x_end']])
    y_top_l = (face['m_top'] * x_arr + face['c_top']).astype(int)
    y_bot_l = (face['m_bot'] * x_arr + face['c_bot']).astype(int)
    cv2.line(vis, (x_arr[0], y_top_l[0]),
             (x_arr[1], y_top_l[1]), (0, 255, 0), 2)
    cv2.line(vis, (x_arr[0], y_bot_l[0]),
             (x_arr[1], y_bot_l[1]), (0, 0, 255), 2)

    x_mid = w // 2
    y_t   = int(face['m_top'] * x_mid + face['c_top'])
    y_b   = int(face['m_bot'] * x_mid + face['c_bot'])
    cv2.arrowedLine(vis, (x_mid, y_t), (x_mid, y_b), (0,255,255), 2)
    cv2.arrowedLine(vis, (x_mid, y_b), (x_mid, y_t), (0,255,255), 2)
    cv2.putText(vis, f"{m['face_width_mm']:.1f}mm",
                (x_mid+10, (y_t+y_b)//2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    if shaft:
        cv2.line(vis,
                 (int(shaft['shaft_x_left']),  0),
                 (int(shaft['shaft_x_left']),  y_t),
                 (255, 0, 255), 2)
        cv2.line(vis,
                 (int(shaft['shaft_x_right']), 0),
                 (int(shaft['shaft_x_right']), y_t),
                 (255, 0, 255), 2)

    if helix.get('diag_lines'):
        roi_top = helix['roi_top']
        x_off   = helix['x_start']
        for x1,y1,x2,y2 in helix['diag_lines']:
            cv2.line(vis,
                     (x1+x_off, y1+roi_top),
                     (x2+x_off, y2+roi_top),
                     (255, 165, 0), 1)

    side_meas = {
        'face_width_mm'      : m['face_width_mm'],
        'perp_deviation_deg' : m.get('perp_deviation_deg') or 0,
        'helix_angle_deg'    : m.get('helix_angle_deg') or 0,
    }
    return side_meas, vis


def check_tolerances(measurements, tolerances):
    results = {}
    overall = 'PASS'
    for key, (lo, hi) in tolerances.items():
        val = measurements.get(key)
        if val is None:
            results[key] = ('N/A', None, lo, hi)
            continue
        status = 'PASS' if lo <= val <= hi else 'FAIL'
        if status == 'FAIL':
            overall = 'FAIL'
        results[key] = (status, val, lo, hi)
    return results, overall


# ══════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD
# ══════════════════════════════════════════════════════════════════════════
class GearDashboard:

    PREVIEW_W = 400    # width of each camera preview panel
    PREVIEW_H = 300    # height of each camera preview panel
    REFRESH   = 100     # ms between frame refreshes (~12 fps)

    def __init__(self, root):
        self.root       = root
        self.root.title("Gear Inspection Dashboard")
        self.root.geometry("1400x860")
        self.root.configure(bg='#1e1e1e')

        self.tolerances = dict(DEFAULT_TOLERANCES)

        # Cameras
        self.cam_top  = IPCamera()   # endface (top-down)
        self.cam_side = IPCamera()   # side profile

        # Captured frames
        self.captured_endface = None
        self.captured_side    = None

        # Result images
        self.result_endface   = None
        self.result_side      = None

        # Photo references (prevent GC)
        self._photos = {}

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._start_preview_loop()

    # ── UI ─────────────────────────────────────────────────────────────────
    def _build_ui(self):

        # Load saved IPs
        self._load_config()

        # ── Top bar ────────────────────────────────────────────────────────
        topbar = tk.Frame(self.root, bg='#2d2d2d', height=52)
        topbar.pack(fill='x', side='top')
        topbar.pack_propagate(False)

        tk.Label(topbar, text="⚙  Gear Inspection System",
                 bg='#2d2d2d', fg='white',
                 font=('Helvetica', 13, 'bold')).pack(
            side='left', padx=14)

        # Camera connect section
        tk.Label(topbar, text="Cam 1 IP:",
                 bg='#2d2d2d', fg='#aaaaaa',
                 font=('Helvetica', 9)).pack(side='left', padx=(16,2))
        self.ip_top = tk.Entry(topbar, width=16,
                               bg='#3d3d3d', fg='white',
                               insertbackground='white',
                               font=('Courier', 9), relief='flat')
        self.ip_top.insert(0, "192.168.x.x")
        self.ip_top.pack(side='left', pady=10)

        self.dot_top = tk.Label(topbar, text="●",
                                bg='#2d2d2d', fg='#555555',
                                font=('Helvetica', 12))
        self.dot_top.pack(side='left', padx=3)

        tk.Label(topbar, text="Cam 2 IP:",
                 bg='#2d2d2d', fg='#aaaaaa',
                 font=('Helvetica', 9)).pack(side='left', padx=(10,2))
        self.ip_side = tk.Entry(topbar, width=16,
                                bg='#3d3d3d', fg='white',
                                insertbackground='white',
                                font=('Courier', 9), relief='flat')
        self.ip_side.insert(0, "192.168.x.x")
        self.ip_side.pack(side='left', pady=10)

        self.dot_side = tk.Label(topbar, text="●",
                                 bg='#2d2d2d', fg='#555555',
                                 font=('Helvetica', 12))
        self.dot_side.pack(side='left', padx=3)

        tk.Button(topbar, text="Connect",
                  command=self._connect_cameras,
                  bg='#0078d4', fg='white',
                  font=('Helvetica', 9), relief='flat',
                  padx=10).pack(side='left', padx=8, pady=10)

        tk.Button(topbar, text="Disconnect",
                  command=self._disconnect_cameras,
                  bg='#555555', fg='white',
                  font=('Helvetica', 9), relief='flat',
                  padx=8).pack(side='left', padx=2, pady=10)

        # Separator
        tk.Frame(topbar, bg='#444444', width=2).pack(
            side='left', fill='y', padx=10, pady=8)

        tk.Button(topbar, text="📷  Capture & Inspect",
                  command=self._capture_and_inspect,
                  bg='#107c10', fg='white',
                  font=('Helvetica', 10, 'bold'), relief='flat',
                  padx=14).pack(side='left', padx=4, pady=8)

        tk.Button(topbar, text="Load Files",
                  command=self._load_files,
                  bg='#5c2d91', fg='white',
                  font=('Helvetica', 9), relief='flat',
                  padx=10).pack(side='left', padx=4, pady=10)

        tk.Button(topbar, text="Tolerances",
                  command=self._open_tolerance_editor,
                  bg='#7a3d00', fg='white',
                  font=('Helvetica', 9), relief='flat',
                  padx=10).pack(side='left', padx=4, pady=10)

        self.status_label = tk.Label(
            topbar, text="Not connected",
            bg='#2d2d2d', fg='#aaaaaa',
            font=('Helvetica', 9))
        self.status_label.pack(side='right', padx=14)

        # ── Main area ──────────────────────────────────────────────────────
        main = tk.Frame(self.root, bg='#1e1e1e')
        main.pack(fill='both', expand=True)

        # Left: camera previews stacked
        left = tk.Frame(main, bg='#1e1e1e', width=self.PREVIEW_W + 16)
        left.pack(side='left', fill='y', padx=8, pady=8)
        left.pack_propagate(False)

        # Top-down camera panel
        self._build_camera_panel(left, "TOP-DOWN  (Endface)",
                                  "cam_top_label", is_top=True)

        # Side profile camera panel
        self._build_camera_panel(left, "SIDE PROFILE",
                                  "cam_side_label", is_top=False)

        # Centre: result images
        centre = tk.Frame(main, bg='#1e1e1e')
        centre.pack(side='left', fill='both', expand=True, pady=8)

        tk.Label(centre, text="Inspection Results",
                 bg='#1e1e1e', fg='#888888',
                 font=('Helvetica', 9)).pack(anchor='w', padx=4)

        self.result_label = tk.Label(centre, bg='#2d2d2d',
                                      text="Results appear here after capture",
                                      fg='#555555',
                                      font=('Helvetica', 10))
        self.result_label.pack(fill='both', expand=True, padx=4)

        # Right: measurements
        right = tk.Frame(main, bg='#1e1e1e', width=340)
        right.pack(side='right', fill='y', padx=8, pady=8)
        right.pack_propagate(False)

        # Banner
        self.banner = tk.Label(right,
                                text="── AWAITING CAPTURE ──",
                                bg='#3d3d3d', fg='white',
                                font=('Helvetica', 12, 'bold'),
                                pady=10)
        self.banner.pack(fill='x', pady=(0, 6))

        # Measurement table
        tk.Label(right, text="Measurements",
                 bg='#1e1e1e', fg='#888888',
                 font=('Helvetica', 9)).pack(anchor='w')

        tbl = tk.Frame(right, bg='#2d2d2d')
        tbl.pack(fill='both', expand=True, pady=(2, 6))

        headers = ['Measurement', 'Value', 'Range', 'OK?']
        col_w   = [13, 7, 9, 5]
        for c, (h, w) in enumerate(zip(headers, col_w)):
            tk.Label(tbl, text=h, bg='#3d3d3d', fg='#cccccc',
                     font=('Helvetica', 8, 'bold'),
                     width=w, anchor='w', padx=4,
                     pady=3).grid(row=0, column=c,
                                  sticky='ew', padx=1, pady=1)

        self.row_defs = [
            ('Tip diameter',    'tip_diameter_mm',    'mm', 2),
            ('Root diameter',   'root_diameter_mm',   'mm', 2),
            ('Pitch diameter',  'pitch_diameter_mm',  'mm', 2),
            ('Shaft diameter',  'shaft_diameter_mm',  'mm', 2),
            ('Tooth count',     'tooth_count',        '',   0),
            ('Tooth width',     'tooth_width_mm',     'mm', 3),
            ('Tooth depth',     'tooth_depth_mm',     'mm', 3),
            ('Circularity',     'circularity',        '',   3),
            ('Face width',      'face_width_mm',      'mm', 2),
            ('Perp deviation',  'perp_deviation_deg', '°',  2),
            ('Helix angle',     'helix_angle_deg',    '°',  2),
        ]

        self.row_labels = {}
        for r, (label, key, unit, dec) in enumerate(self.row_defs, 1):
            bg = '#2a2a2a' if r % 2 == 0 else '#252525'
            tk.Label(tbl, text=label, bg=bg, fg='#dddddd',
                     font=('Helvetica', 8), anchor='w',
                     padx=4, pady=4).grid(
                row=r, column=0, sticky='ew', padx=1, pady=1)
            v = tk.Label(tbl, text='—', bg=bg, fg='white',
                         font=('Helvetica', 8, 'bold'), anchor='e', padx=4)
            v.grid(row=r, column=1, sticky='ew', padx=1, pady=1)
            rn = tk.Label(tbl, text='—', bg=bg, fg='#888888',
                          font=('Helvetica', 7), anchor='center', padx=2)
            rn.grid(row=r, column=2, sticky='ew', padx=1, pady=1)
            st = tk.Label(tbl, text='—', bg=bg, fg='#888888',
                          font=('Helvetica', 8, 'bold'), anchor='center')
            st.grid(row=r, column=3, sticky='ew', padx=1, pady=1)
            self.row_labels[key] = (v, rn, st, bg, unit, dec)

        # Log
        log_f = tk.Frame(right, bg='#2d2d2d')
        log_f.pack(fill='x', pady=2)
        self.log_txt = tk.Text(log_f, height=4,
                                bg='#2d2d2d', fg='#aaaaaa',
                                font=('Courier', 7),
                                relief='flat', padx=6, pady=4)
        self.log_txt.pack(fill='x')
        self.log_txt.insert('end', 'Pipeline log will appear here.')
        self.log_txt.config(state='disabled')

    def _build_camera_panel(self, parent, title, attr, is_top):
        """Builds a labelled camera preview panel with fixed size."""
        frame = tk.Frame(parent, bg='#1e1e1e')
        frame.pack(fill='x', pady=(0, 6))

        tk.Label(frame, text=title,
                 bg='#1e1e1e', fg='#888888',
                 font=('Helvetica', 8)).pack(anchor='w')

        # Fixed-size container — prevents label from shrinking
        container = tk.Frame(frame, bg='#2d2d2d',
                              width=self.PREVIEW_W,
                              height=self.PREVIEW_H)
        container.pack()
        container.pack_propagate(False)   # ← holds fixed size

        lbl = tk.Label(container, bg='#2d2d2d',
                        text="Not connected",
                        fg='#555555',
                        font=('Helvetica', 9))
        lbl.place(relx=0.5, rely=0.5, anchor='center')
        setattr(self, attr, lbl)
    
    # ── Camera connection ──────────────────────────────────────────────────
    def _connect_cameras(self):
        self.status_label.config(text="Connecting…", fg='#ffb900')
        self.root.update()

        def worker():
            ip1   = self.ip_top.get().strip()
            ip2   = self.ip_side.get().strip()
            ok1   = self.cam_top.connect(ip1)
            ok2   = self.cam_side.connect(ip2)

            def update():
                self.dot_top.config(
                    fg='#16c60c' if ok1 else '#e81123')
                self.dot_side.config(
                    fg='#16c60c' if ok2 else '#e81123')

                if ok1 and ok2:
                    msg = "Both cameras connected"
                    col = '#16c60c'
                elif ok1 or ok2:
                    msg = "One camera connected"
                    col = '#ffb900'
                else:
                    msg = "Connection failed — check IPs"
                    col = '#e81123'

                self.status_label.config(text=msg, fg=col)
                self._log(
                    f"Cam1 (top-down)  : {'OK' if ok1 else 'FAILED'}\n"
                    f"Cam2 (side)      : {'OK' if ok2 else 'FAILED'}\n"
                    f"Stream: IP:8080/video\n"
                    f"Snapshot: IP:8080/shot.jpg")

            self.root.after(0, update)

        threading.Thread(target=worker, daemon=True).start()

    def _disconnect_cameras(self):
        self.cam_top.disconnect()
        self.cam_side.disconnect()
        self.dot_top.config(fg='#555555')
        self.dot_side.config(fg='#555555')
        self.status_label.config(text="Disconnected", fg='#aaaaaa')

    # ── Live preview loop ──────────────────────────────────────────────────
    def _start_preview_loop(self):
        self._update_previews()

    def _update_previews(self):
        """Refreshes both camera preview panels at REFRESH interval."""
        self._refresh_one(self.cam_top,  self.cam_top_label,
                          endface=True)
        self._refresh_one(self.cam_side, self.cam_side_label,
                          endface=False)
        self.root.after(self.REFRESH, self._update_previews)

    def _refresh_one(self, cam, label, endface):
        frame = cam.get_frame()
        if frame is None:
            return

        # Resize to preview dimensions in this call
        # (already on UI thread but frame is small so fast)
        h, w = frame.shape[:2]
        # Maintain aspect ratio within preview box
        scale  = min(self.PREVIEW_W / w, self.PREVIEW_H / h)
        nw     = int(w * scale)
        nh     = int(h * scale)
        frame  = cv2.resize(frame, (nw, nh),
                             interpolation=cv2.INTER_LINEAR)

        # Position check
        if endface:
            in_zone = check_endface_position(frame)
            frame   = draw_endface_overlay(frame, in_zone)
        else:
            in_zone = check_sideprofile_position(frame)
            frame   = draw_sideprofile_overlay(frame, in_zone)

        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        label.config(image=photo, text='')
        label.place(relx=0.5, rely=0.5, anchor='center')
        key = 'top' if endface else 'side'
        self._photos[key] = photo
    
    def _load_config(self):
        """Loads saved IPs from config file on startup."""
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r') as f:
                    cfg = json.load(f)
                ip1 = cfg.get('ip_top', '')
                ip2 = cfg.get('ip_side', '')
                if ip1:
                    self.ip_top.delete(0, 'end')
                    self.ip_top.insert(0, ip1)
                if ip2:
                    self.ip_side.delete(0, 'end')
                    self.ip_side.insert(0, ip2)
        except Exception:
            pass   # silently ignore corrupt config

    def _save_config(self):
        """Saves current IPs to config file on exit."""
        try:
            cfg = {
                'ip_top'  : self.ip_top.get().strip(),
                'ip_side' : self.ip_side.get().strip(),
            }
            with open(CONFIG_FILE, 'w') as f:
                json.dump(cfg, f, indent=2)
        except Exception:
            pass

    # ── Capture & inspect ──────────────────────────────────────────────────
    def _capture_and_inspect(self):
        """Captures from both cameras then runs inspection automatically."""

        # Check at least one camera is connected
        if not self.cam_top.connected and not self.cam_side.connected:
            # Fallback — ask to load files
            ans = messagebox.askyesno(
                "No cameras connected",
                "No cameras are connected.\n\n"
                "Click Yes to load image files manually instead.")
            if ans:
                self._load_files()
            return

        self.banner.config(text="── CAPTURING… ──",
                           bg='#5c2d91', fg='white')
        self.status_label.config(text="Capturing…", fg='#ffb900')
        self.root.update()

        def worker():
            errors = []
            os.makedirs("results", exist_ok=True)

            # Capture endface
            endface_path = None
            if self.cam_top.connected:
                frame = self.cam_top.capture_snapshot()
                if frame is not None:
                    cv2.imwrite(CAPTURE_ENDFACE, frame)
                    endface_path = CAPTURE_ENDFACE
                    self.root.after(0, lambda:
                        self.status_label.config(
                            text="Endface captured…", fg='#ffb900'))
                else:
                    errors.append("Endface capture failed")
            else:
                errors.append("Endface camera not connected")

            # Capture side profile
            side_path = None
            if self.cam_side.connected:
                frame = self.cam_side.capture_snapshot()
                if frame is not None:
                    cv2.imwrite(CAPTURE_SIDE, frame)
                    side_path = CAPTURE_SIDE
                    self.root.after(0, lambda:
                        self.status_label.config(
                            text="Both captured — running pipeline…",
                            fg='#ffb900'))
                else:
                    errors.append("Side profile capture failed")
            else:
                errors.append("Side camera not connected")

            # Run pipeline
            self.root.after(0, lambda:
                self._run_pipeline(endface_path, side_path, errors))

        threading.Thread(target=worker, daemon=True).start()

    def _run_pipeline(self, endface_path, side_path, capture_errors):
        """Runs both pipelines and displays combined results."""

        def worker():
            all_meas     = {}
            result_imgs  = []
            errors       = list(capture_errors)

            # Endface pipeline
            if endface_path and os.path.exists(endface_path):
                try:
                    meas, vis = run_endface_pipeline(endface_path)
                    all_meas.update(meas)
                    result_imgs.append(('Endface', vis))
                except Exception as e:
                    errors.append(f"Endface pipeline: {e}")

            # Side profile pipeline
            if side_path and os.path.exists(side_path):
                try:
                    meas, vis = run_side_pipeline(side_path)
                    all_meas.update(meas)
                    result_imgs.append(('Side', vis))
                except Exception as e:
                    errors.append(f"Side pipeline: {e}")

            if not all_meas:
                self.root.after(0, lambda:
                    self._show_error("Pipeline failed — " +
                                     "; ".join(errors)))
                return

            tol_results, overall = check_tolerances(
                all_meas, self.tolerances)

            self.root.after(0, lambda:
                self._display_results(
                    all_meas, tol_results, overall,
                    result_imgs, errors))

        threading.Thread(target=worker, daemon=True).start()

    def _display_results(self, meas, tol_results,
                          overall, result_imgs, errors):
        """Updates measurement table, banner, and result image."""

        # Banner
        if overall == 'PASS':
            self.banner.config(
                text="✓  PASS — All measurements within tolerance",
                bg='#107c10', fg='white')
        else:
            self.banner.config(
                text="✗  FAIL — Out of tolerance",
                bg='#e81123', fg='white')

        # Measurement rows
        for label, key, unit, dec in self.row_defs:
            v_lbl, rn_lbl, st_lbl, bg, u, d = self.row_labels[key]
            if key in tol_results:
                status, val, lo, hi = tol_results[key]
                v_lbl.config(
                    text=f'{val:.{d}f}{u}' if val is not None else '—')
                lo_s = f'{lo:.{d}f}' if isinstance(lo, float) else str(lo)
                hi_s = f'{hi:.{d}f}' if isinstance(hi, float) else str(hi)
                rn_lbl.config(text=f'{lo_s}–{hi_s}{u}')
                if status == 'PASS':
                    st_lbl.config(text='✓', fg='#16c60c')
                elif status == 'FAIL':
                    st_lbl.config(text='✗', fg='#e81123')
                else:
                    st_lbl.config(text='—', fg='#888888')

        # Combined result image — stitch both annotated images side by side
        if result_imgs:
            combined = self._stitch_results(result_imgs)
            self._show_result_image(combined)

        # Log
        log_lines = []
        if errors:
            log_lines.append("WARNINGS: " + " | ".join(errors))
        log_lines.append(f"Overall: {overall}")
        if 'tip_diameter_mm' in meas:
            log_lines.append(
                f"Tip: {meas['tip_diameter_mm']:.2f}mm  "
                f"Root: {meas['root_diameter_mm']:.2f}mm  "
                f"Teeth: {meas.get('tooth_count','?')}")
        if 'face_width_mm' in meas:
            log_lines.append(
                f"Face: {meas['face_width_mm']:.2f}mm  "
                f"Helix: {meas.get('helix_angle_deg',0):.1f}°")
        self._log("\n".join(log_lines))

        self.status_label.config(
            text=f"Done — {overall}",
            fg='#16c60c' if overall == 'PASS' else '#e81123')

    def _stitch_results(self, result_imgs):
        """Stitches result images side by side with labels."""
        if len(result_imgs) == 1:
            return result_imgs[0][1]

        imgs = []
        for title, img in result_imgs:
            # Add title bar to each image
            h, w = img.shape[:2]
            bar  = np.zeros((30, w, 3), dtype=np.uint8)
            bar[:] = (45, 45, 45)
            cv2.putText(bar, title, (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255,255,255), 1)
            imgs.append(np.vstack([bar, img]))

        # Resize both to same height
        h0 = imgs[0].shape[0]
        resized = []
        for img in imgs:
            h, w = img.shape[:2]
            nw   = int(w * h0 / h)
            resized.append(cv2.resize(img, (nw, h0)))

        return np.hstack(resized)

    def _show_result_image(self, bgr_img):
        self.root.update_idletasks()
        pw = self.result_label.winfo_width()
        ph = self.result_label.winfo_height()
        if pw < 10: pw = 600
        if ph < 10: ph = 760

        h, w   = bgr_img.shape[:2]
        scale  = min(pw/w, ph/h)
        nw, nh = int(w*scale), int(h*scale)
        resized = cv2.resize(bgr_img, (nw, nh))
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        photo   = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.result_label.config(image=photo, text='')
        self._photos['result'] = photo

    def _show_error(self, msg):
        self.banner.config(text=f"ERROR: {msg}",
                           bg='#e81123', fg='white')
        self.status_label.config(text="Error", fg='#e81123')
        self._log(f"ERROR: {msg}")

    # ── File fallback ──────────────────────────────────────────────────────
    def _load_files(self):
        """Load endface and side profile images from disk."""
        ef = filedialog.askopenfilename(
            title="Select ENDFACE image",
            filetypes=[("Images","*.png *.jpg *.jpeg *.bmp")])
        if not ef:
            return
        sp = filedialog.askopenfilename(
            title="Select SIDE PROFILE image",
            filetypes=[("Images","*.png *.jpg *.jpeg *.bmp")])

        self.banner.config(text="── PROCESSING… ──",
                           bg='#5c2d91', fg='white')
        self._run_pipeline(ef, sp or None, [])

    # ── Misc helpers ───────────────────────────────────────────────────────
    def _log(self, text):
        self.log_txt.config(state='normal')
        self.log_txt.delete('1.0', 'end')
        self.log_txt.insert('end', text)
        self.log_txt.config(state='disabled')

    def _open_tolerance_editor(self):
        win = tk.Toplevel(self.root)
        win.title("Edit Tolerances")
        win.geometry("420x420")
        win.configure(bg='#1e1e1e')
        win.grab_set()

        tk.Label(win, text="Set tolerance ranges",
                 bg='#1e1e1e', fg='#aaaaaa',
                 font=('Helvetica', 9)).pack(pady=8)

        frame = tk.Frame(win, bg='#1e1e1e')
        frame.pack(fill='both', expand=True, padx=16)

        for c, h in enumerate(['Measurement','Min','Max']):
            tk.Label(frame, text=h, bg='#2d2d2d', fg='#cccccc',
                     font=('Helvetica', 9,'bold'),
                     padx=8, pady=4).grid(
                row=0, column=c, sticky='ew', padx=2, pady=2)

        entries   = {}
        row_info  = [
            ('Tip diameter mm',    'tip_diameter_mm'),
            ('Root diameter mm',   'root_diameter_mm'),
            ('Pitch diameter mm',  'pitch_diameter_mm'),
            ('Shaft diameter mm',  'shaft_diameter_mm'),
            ('Tooth count',        'tooth_count'),
            ('Tooth width mm',     'tooth_width_mm'),
            ('Tooth depth mm',     'tooth_depth_mm'),
            ('Circularity',        'circularity'),
            ('Face width mm',      'face_width_mm'),
            ('Perp deviation °',   'perp_deviation_deg'),
            ('Helix angle °',      'helix_angle_deg'),
        ]

        for r, (label, key) in enumerate(row_info, 1):
            bg   = '#2a2a2a' if r%2==0 else '#252525'
            lo, hi = self.tolerances.get(key, (0, 999))
            tk.Label(frame, text=label, bg=bg, fg='#dddddd',
                     font=('Helvetica', 9), anchor='w',
                     padx=6, pady=3).grid(
                row=r, column=0, sticky='ew', padx=2, pady=1)
            lo_v = tk.StringVar(value=str(lo))
            hi_v = tk.StringVar(value=str(hi))
            tk.Entry(frame, textvariable=lo_v, width=8,
                     bg='#3d3d3d', fg='white',
                     insertbackground='white',
                     relief='flat').grid(row=r, column=1, padx=4, pady=1)
            tk.Entry(frame, textvariable=hi_v, width=8,
                     bg='#3d3d3d', fg='white',
                     insertbackground='white',
                     relief='flat').grid(row=r, column=2, padx=4, pady=1)
            entries[key] = (lo_v, hi_v)

        def save():
            for key, (lo_v, hi_v) in entries.items():
                try:
                    self.tolerances[key] = (
                        float(lo_v.get()), float(hi_v.get()))
                except ValueError:
                    pass
            win.destroy()

        tk.Button(win, text="Save Tolerances",
                  command=save,
                  bg='#0078d4', fg='white',
                  font=('Helvetica', 10), relief='flat',
                  padx=12, pady=6).pack(pady=10)
    
    def _on_close(self):
        """Saves config and closes cleanly."""
        self._save_config()
        self._disconnect_cameras()
        self.root.destroy()

# ══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════
def main():
    root = tk.Tk()
    app  = GearDashboard(root)
    root.mainloop()


if __name__ == '__main__':
    main()