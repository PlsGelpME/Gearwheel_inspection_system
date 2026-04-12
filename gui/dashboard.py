import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import threading
import os

from pipeline.gear_core      import run_core_pipeline
from pipeline.gear_mask      import build_gear_mask
from pipeline.tooth_analysis import (get_contour_signal,
                                      detect_teeth_from_contour,
                                      measure_teeth_from_contour)


# ══════════════════════════════════════════════════════════════════════════
# DEFAULT TOLERANCES  (edit these to match your gear specification)
# ══════════════════════════════════════════════════════════════════════════
DEFAULT_TOLERANCES = {
    'tip_diameter_mm'   : (39.0,  41.0),   # (min, max)
    'root_diameter_mm'  : (37.5,  39.0),
    'pitch_diameter_mm' : (38.5,  40.0),
    'shaft_diameter_mm' : (2.8,   3.3),
    'tooth_count'       : (71,    71),
    'tooth_width_mm'    : (1.4,   2.0),
    'tooth_depth_mm'    : (0.7,   1.2),
    'circularity'       : (0.75,  1.0),
}

PX_PER_MM      = 20.0
EXPECTED_TEETH = 71


# ══════════════════════════════════════════════════════════════════════════
# MEASUREMENT RUNNER
# ══════════════════════════════════════════════════════════════════════════
def run_full_pipeline(image_path):
    """
    Runs the full measurement pipeline on one image.
    Returns (measurements_dict, annotated_image_bgr).
    """
    m, inter = run_core_pipeline(image_path, px_per_mm=PX_PER_MM)

    gray    = inter['gray']
    gcx     = m['gear_centre_x']
    gcy     = m['gear_centre_y']
    tip_r   = inter['tip_radius_px']
    root_r  = inter['root_radius_px']
    shaft_r = inter['shaft_radius_px']

    # Tooth analysis
    gear_only, _ = build_gear_mask(gray, gcx, gcy, shaft_r, tip_r)
    angles, dists, pts = get_contour_signal(
        gear_only, gcx, gcy, tip_radius_px=tip_r)
    tooth_peaks, gap_valleys, smoothed = detect_teeth_from_contour(
        dists, angles, expected_count=EXPECTED_TEETH)
    tooth_meas, summary = measure_teeth_from_contour(
        dists, angles, pts,
        tooth_peaks, gap_valleys, smoothed,
        gcx, gcy, PX_PER_MM)

    tip_r_fit  = summary.get('tip_radius_fitted_px',  tip_r)
    root_r_fit = summary.get('root_radius_fitted_px', root_r)
    pitch_r    = tip_r_fit - (tip_r_fit - root_r_fit) / 2.25

    # Assemble measurements
    measurements = {
        'tip_diameter_mm'   : tip_r_fit  * 2 / PX_PER_MM,
        'root_diameter_mm'  : root_r_fit * 2 / PX_PER_MM,
        'pitch_diameter_mm' : pitch_r    * 2 / PX_PER_MM,
        'shaft_diameter_mm' : m['shaft_diameter_px'] / PX_PER_MM,
        'tooth_count'       : summary.get('tooth_count', 0),
        'tooth_width_mm'    : summary.get('tooth_width_mean_mm', 0),
        'tooth_depth_mm'    : summary.get('tooth_depth_mean_mm', 0),
        'circularity'       : m['circularity'],
        'circularity_gdt_mm': m.get('circularity_gdt_zone', 0) / PX_PER_MM,
        'gear_cx'           : gcx,
        'gear_cy'           : gcy,

        # Raw px values for drawing
        '_tip_r'   : tip_r_fit,
        '_root_r'  : root_r_fit,
        '_pitch_r' : pitch_r,
        '_shaft_r' : m['shaft_diameter_px'] / 2,
        '_shaft_ellipse': inter.get('shaft_ellipse'),
    }

    # Build annotated image
    annotated = _annotate(gray, measurements, tooth_meas[:5])

    return measurements, annotated


def _annotate(gray, meas, first5_teeth):
    """Draws measurement overlays on the image."""
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    gcx = meas['gear_cx']
    gcy = meas['gear_cy']

    # Circles
    cv2.circle(vis, (gcx, gcy),
               int(meas['_tip_r']),   (0, 140, 255), 2)   # orange = tip
    cv2.circle(vis, (gcx, gcy),
               int(meas['_root_r']),  (0,   0, 255), 2)   # red    = root
    cv2.circle(vis, (gcx, gcy),
               int(meas['_pitch_r']), (0, 255,   0), 2)   # green  = pitch
    cv2.circle(vis, (gcx, gcy),
               int(meas['_shaft_r']), (255, 255, 0), 2)   # cyan   = shaft

    # Gear centre crosshair
    cv2.line(vis, (gcx-40, gcy), (gcx+40, gcy), (255,255,255), 1)
    cv2.line(vis, (gcx, gcy-40), (gcx, gcy+40), (255,255,255), 1)

    # Annotate first 5 teeth
    for mt in first5_teeth:
        tip_pt   = (int(mt['tip_x']),   int(mt['tip_y']))
        left_pt  = (int(mt['left_x']),  int(mt['left_y']))
        right_pt = (int(mt['right_x']), int(mt['right_y']))
        cv2.circle(vis, tip_pt,   3, (0,   0, 255), -1)
        cv2.circle(vis, left_pt,  3, (0, 255,   0), -1)
        cv2.circle(vis, right_pt, 3, (255, 80,  0), -1)
        cv2.line(vis, left_pt, right_pt, (255,255,0), 1)

    return vis


def check_tolerances(measurements, tolerances):
    """
    Compares measurements against tolerances.
    Returns dict: key → ('PASS'|'FAIL'|'WARN', value, min, max)
    """
    results  = {}
    overall  = 'PASS'

    for key, (lo, hi) in tolerances.items():
        val = measurements.get(key)
        if val is None:
            results[key] = ('N/A', None, lo, hi)
            continue

        if lo <= val <= hi:
            status = 'PASS'
        else:
            status  = 'FAIL'
            overall = 'FAIL'

        results[key] = (status, val, lo, hi)

    return results, overall


# ══════════════════════════════════════════════════════════════════════════
# GUI
# ══════════════════════════════════════════════════════════════════════════
class GearDashboard:

    def __init__(self, root):
        self.root        = root
        self.root.title("Gear Inspection Dashboard")
        self.root.geometry("1280x800")
        self.root.configure(bg='#1e1e1e')

        self.tolerances  = dict(DEFAULT_TOLERANCES)
        self.image_path  = None
        self.photo       = None

        self._build_ui()

    # ── UI Layout ──────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Top bar ────────────────────────────────────────────────────────
        topbar = tk.Frame(self.root, bg='#2d2d2d', height=50)
        topbar.pack(fill='x', side='top')
        topbar.pack_propagate(False)

        tk.Label(topbar, text="⚙  Gear Inspection System",
                 bg='#2d2d2d', fg='white',
                 font=('Helvetica', 14, 'bold')).pack(side='left', padx=16)

        tk.Button(topbar, text="Load Image",
                  command=self._load_image,
                  bg='#0078d4', fg='white',
                  font=('Helvetica', 10),
                  relief='flat', padx=12).pack(side='left', padx=8, pady=8)

        tk.Button(topbar, text="Run Inspection",
                  command=self._run_inspection,
                  bg='#107c10', fg='white',
                  font=('Helvetica', 10),
                  relief='flat', padx=12).pack(side='left', padx=4, pady=8)

        tk.Button(topbar, text="Tolerances",
                  command=self._open_tolerance_editor,
                  bg='#5c2d91', fg='white',
                  font=('Helvetica', 10),
                  relief='flat', padx=12).pack(side='left', padx=4, pady=8)

        self.status_label = tk.Label(
            topbar, text="No image loaded",
            bg='#2d2d2d', fg='#aaaaaa',
            font=('Helvetica', 10))
        self.status_label.pack(side='right', padx=16)

        # ── Main area ──────────────────────────────────────────────────────
        main = tk.Frame(self.root, bg='#1e1e1e')
        main.pack(fill='both', expand=True)

        # Left: image panel
        left = tk.Frame(main, bg='#1e1e1e')
        left.pack(side='left', fill='both', expand=True, padx=8, pady=8)

        tk.Label(left, text="Image", bg='#1e1e1e', fg='#888888',
                 font=('Helvetica', 9)).pack(anchor='w')

        self.image_label = tk.Label(left, bg='#2d2d2d',
                                     text="Load an image to begin",
                                     fg='#555555',
                                     font=('Helvetica', 12))
        self.image_label.pack(fill='both', expand=True)

        # Right: measurements panel
        right = tk.Frame(main, bg='#1e1e1e', width=380)
        right.pack(side='right', fill='y', padx=8, pady=8)
        right.pack_propagate(False)

        # Overall result banner
        self.banner = tk.Label(right,
                                text="── AWAITING INSPECTION ──",
                                bg='#3d3d3d', fg='white',
                                font=('Helvetica', 14, 'bold'),
                                pady=12)
        self.banner.pack(fill='x', pady=(0, 8))

        # Measurements table
        tk.Label(right, text="Measurements",
                 bg='#1e1e1e', fg='#888888',
                 font=('Helvetica', 9)).pack(anchor='w')

        table_frame = tk.Frame(right, bg='#2d2d2d')
        table_frame.pack(fill='both', expand=True, pady=(4, 8))

        # Table headers
        headers = ['Measurement', 'Value', 'Range', 'Status']
        col_w   = [160, 70, 100, 60]
        for c, (h, w) in enumerate(zip(headers, col_w)):
            tk.Label(table_frame, text=h,
                     bg='#3d3d3d', fg='#cccccc',
                     font=('Helvetica', 9, 'bold'),
                     width=w//7, anchor='w',
                     padx=6, pady=4).grid(
                row=0, column=c, sticky='ew', padx=1, pady=1)

        # Measurement row definitions
        self.row_defs = [
            ('Tip diameter',    'tip_diameter_mm',    'mm', 2),
            ('Root diameter',   'root_diameter_mm',   'mm', 2),
            ('Pitch diameter',  'pitch_diameter_mm',  'mm', 2),
            ('Shaft diameter',  'shaft_diameter_mm',  'mm', 2),
            ('Tooth count',     'tooth_count',        '',   0),
            ('Tooth width',     'tooth_width_mm',     'mm', 3),
            ('Tooth depth',     'tooth_depth_mm',     'mm', 3),
            ('Circularity ISO', 'circularity',        '',   3),
            ('Roundness GD&T',  'circularity_gdt_mm', 'mm', 3),
        ]

        self.row_labels = {}
        for r, (label, key, unit, dec) in enumerate(self.row_defs, start=1):
            bg = '#2a2a2a' if r % 2 == 0 else '#252525'

            tk.Label(table_frame, text=label,
                     bg=bg, fg='#dddddd',
                     font=('Helvetica', 9),
                     anchor='w', padx=6, pady=5).grid(
                row=r, column=0, sticky='ew', padx=1, pady=1)

            val_lbl = tk.Label(table_frame, text='—',
                               bg=bg, fg='#ffffff',
                               font=('Helvetica', 9, 'bold'),
                               anchor='e', padx=6)
            val_lbl.grid(row=r, column=1, sticky='ew', padx=1, pady=1)

            rng_lbl = tk.Label(table_frame, text='—',
                               bg=bg, fg='#888888',
                               font=('Helvetica', 8),
                               anchor='center', padx=4)
            rng_lbl.grid(row=r, column=2, sticky='ew', padx=1, pady=1)

            st_lbl = tk.Label(table_frame, text='—',
                              bg=bg, fg='#888888',
                              font=('Helvetica', 9, 'bold'),
                              anchor='center')
            st_lbl.grid(row=r, column=3, sticky='ew', padx=1, pady=1)

            self.row_labels[key] = (val_lbl, rng_lbl, st_lbl, bg, unit, dec)

        # Info panel below table
        info_frame = tk.Frame(right, bg='#2d2d2d')
        info_frame.pack(fill='x', pady=4)

        self.info_text = tk.Text(info_frame, height=5,
                                  bg='#2d2d2d', fg='#aaaaaa',
                                  font=('Courier', 8),
                                  relief='flat', padx=8, pady=6)
        self.info_text.pack(fill='x')
        self.info_text.insert('end', 'Pipeline output will appear here.')
        self.info_text.config(state='disabled')

    # ── Actions ────────────────────────────────────────────────────────────
    def _load_image(self):
        path = filedialog.askopenfilename(
            title="Select gear image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp"),
                       ("All files", "*.*")])
        if not path:
            return

        self.image_path = path
        self.status_label.config(
            text=f"Loaded: {os.path.basename(path)}", fg='#aaaaaa')

        # Show raw image immediately
        img = cv2.imread(path)
        self._show_image(img)
        self.banner.config(text="── IMAGE LOADED — RUN INSPECTION ──",
                           bg='#3d3d3d', fg='white')
        self._log(f"Loaded: {path}")

    def _run_inspection(self):
        if not self.image_path:
            self.status_label.config(
                text="Load an image first", fg='#e81123')
            return

        self.banner.config(text="── PROCESSING… ──",
                           bg='#5c2d91', fg='white')
        self.status_label.config(text="Running pipeline…", fg='#ffb900')
        self.root.update()

        # Run in thread to keep UI responsive
        def worker():
            try:
                meas, annotated = run_full_pipeline(self.image_path)
                tol_results, overall = check_tolerances(
                    meas, self.tolerances)
                self.root.after(0, lambda: self._display_results(
                    meas, tol_results, overall, annotated))
            except Exception as e:
                self.root.after(0, lambda: self._show_error(str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _display_results(self, meas, tol_results, overall, annotated):
        """Updates all UI elements with measurement results."""

        # Overall banner
        if overall == 'PASS':
            self.banner.config(
                text="✓  PASS — All measurements within tolerance",
                bg='#107c10', fg='white')
        else:
            self.banner.config(
                text="✗  FAIL — One or more measurements out of tolerance",
                bg='#e81123', fg='white')

        # Update each row
        for label, key, unit, dec in self.row_defs:
            val_lbl, rng_lbl, st_lbl, bg, u, d = self.row_labels[key]

            if key in tol_results:
                status, val, lo, hi = tol_results[key]

                # Value
                if val is not None:
                    fmt = f'{val:.{d}f}{u}'
                else:
                    fmt = '—'
                val_lbl.config(text=fmt)

                # Range
                lo_str = f'{lo:.{d}f}' if isinstance(lo, float) else str(lo)
                hi_str = f'{hi:.{d}f}' if isinstance(hi, float) else str(hi)
                rng_lbl.config(text=f'{lo_str}–{hi_str}{u}')

                # Status colour
                if status == 'PASS':
                    st_lbl.config(text='PASS', fg='#16c60c')
                elif status == 'FAIL':
                    st_lbl.config(text='FAIL', fg='#e81123')
                else:
                    st_lbl.config(text='N/A',  fg='#888888')

            elif key == 'circularity_gdt_mm':
                # Not in tolerances dict — just show value
                val = meas.get(key)
                if val is not None:
                    val_lbl.config(text=f'{val:.3f}mm')
                    rng_lbl.config(text='info only')
                    st_lbl.config(text='—', fg='#888888')

        # Show annotated image
        self._show_image(annotated)

        # Info log
        self._log(
            f"Gear centre    : ({meas['gear_cx']}, {meas['gear_cy']})\n"
            f"px per mm      : {PX_PER_MM}\n"
            f"Tip diam (px)  : {meas['_tip_r']*2:.1f}\n"
            f"Root diam (px) : {meas['_root_r']*2:.1f}\n"
            f"Overall        : {overall}"
        )

        self.status_label.config(
            text=f"Done — {overall}", fg='#16c60c' if overall=='PASS' else '#e81123')

    def _show_image(self, bgr_img):
        """Resizes and displays a BGR image in the image panel."""
        # Get panel size
        self.root.update_idletasks()
        pw = self.image_label.winfo_width()
        ph = self.image_label.winfo_height()
        if pw < 10 or ph < 10:
            pw, ph = 860, 760

        h, w = bgr_img.shape[:2]
        scale = min(pw / w, ph / h)
        nw, nh = int(w * scale), int(h * scale)

        resized = cv2.resize(bgr_img, (nw, nh))
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        self.photo = ImageTk.PhotoImage(pil_img)

        self.image_label.config(image=self.photo, text='')

    def _show_error(self, msg):
        self.banner.config(text=f"ERROR: {msg}",
                           bg='#e81123', fg='white')
        self.status_label.config(text="Error", fg='#e81123')
        self._log(f"ERROR: {msg}")

    def _log(self, text):
        self.info_text.config(state='normal')
        self.info_text.delete('1.0', 'end')
        self.info_text.insert('end', text)
        self.info_text.config(state='disabled')

    # ── Tolerance editor ───────────────────────────────────────────────────
    def _open_tolerance_editor(self):
        win = tk.Toplevel(self.root)
        win.title("Edit Tolerances")
        win.geometry("420x380")
        win.configure(bg='#1e1e1e')
        win.grab_set()

        tk.Label(win, text="Set tolerance ranges for each measurement",
                 bg='#1e1e1e', fg='#aaaaaa',
                 font=('Helvetica', 9)).pack(pady=8)

        frame = tk.Frame(win, bg='#1e1e1e')
        frame.pack(fill='both', expand=True, padx=16)

        headers = ['Measurement', 'Min', 'Max']
        for c, h in enumerate(headers):
            tk.Label(frame, text=h,
                     bg='#2d2d2d', fg='#cccccc',
                     font=('Helvetica', 9, 'bold'),
                     padx=8, pady=4).grid(
                row=0, column=c, sticky='ew', padx=2, pady=2)

        entries = {}
        row_info = [
            ('Tip diameter mm',   'tip_diameter_mm'),
            ('Root diameter mm',  'root_diameter_mm'),
            ('Pitch diameter mm', 'pitch_diameter_mm'),
            ('Shaft diameter mm', 'shaft_diameter_mm'),
            ('Tooth count',       'tooth_count'),
            ('Tooth width mm',    'tooth_width_mm'),
            ('Tooth depth mm',    'tooth_depth_mm'),
            ('Circularity',       'circularity'),
        ]

        for r, (label, key) in enumerate(row_info, start=1):
            bg = '#2a2a2a' if r % 2 == 0 else '#252525'
            lo, hi = self.tolerances.get(key, (0, 999))

            tk.Label(frame, text=label,
                     bg=bg, fg='#dddddd',
                     font=('Helvetica', 9),
                     anchor='w', padx=6, pady=4).grid(
                row=r, column=0, sticky='ew', padx=2, pady=1)

            lo_var = tk.StringVar(value=str(lo))
            hi_var = tk.StringVar(value=str(hi))

            tk.Entry(frame, textvariable=lo_var, width=8,
                     bg='#3d3d3d', fg='white',
                     insertbackground='white',
                     relief='flat').grid(
                row=r, column=1, padx=4, pady=1)

            tk.Entry(frame, textvariable=hi_var, width=8,
                     bg='#3d3d3d', fg='white',
                     insertbackground='white',
                     relief='flat').grid(
                row=r, column=2, padx=4, pady=1)

            entries[key] = (lo_var, hi_var)

        def save():
            for key, (lo_var, hi_var) in entries.items():
                try:
                    lo = float(lo_var.get())
                    hi = float(hi_var.get())
                    self.tolerances[key] = (lo, hi)
                except ValueError:
                    pass
            win.destroy()

        tk.Button(win, text="Save Tolerances",
                  command=save,
                  bg='#0078d4', fg='white',
                  font=('Helvetica', 10),
                  relief='flat', padx=12,
                  pady=6).pack(pady=12)


# ══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════
def main():
    root = tk.Tk()
    app  = GearDashboard(root)
    root.mainloop()


if __name__ == '__main__':
    main()