import os
import sys
import subprocess
import warnings
from pathlib import Path

# Silence deprecation notices to keep terminal outputs clear
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------
# AUTOMATIC DEPENDENCY CHECKER & INSTALLER
# ---------------------------------------------------------
def _ensure_dependencies():
    """Checks requirements.txt and installs missing packages dynamically."""
    project_root = Path(os.path.dirname(os.path.abspath(__file__)))
    req_path = project_root.parent / "requirements.txt"
    
    try:
        import tabulate
    except ImportError:
        print("[INFO] Installing 'tabulate' for Pandas markdown generation...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tabulate"])

    if req_path.exists():
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", str(req_path)])
        except subprocess.CalledProcessError:
            sys.exit(1)

_ensure_dependencies()

import cv2
import numpy as np
import supervision as sv
import pandas as pd
from ultralytics import YOLO

# ---------------------------------------------------------
# PERFORMANCE CONFIGURATION
# ---------------------------------------------------------
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720
CONFIDENCE_THRESHOLD = 0.25  
YOLO_INTERNAL_RESOLUTION = 640  
FRAME_SKIP = 2                  

LANE_WIDTH = 3.5
METER_SEGMENT_DISTANCE = 1.0
HEADWAY_REFERENCE_METER = 3

# --- CROP CONFIGURATION ---
CROP_X_MIN = 335
CROP_Y_MIN = 169
CROP_X_MAX = FRAME_WIDTH
CROP_Y_MAX = FRAME_HEIGHT

PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = PROJECT_ROOT.parent.parent / "00_weights"
MODEL_PATH = str(WEIGHTS_DIR / "best.pt") 

PED_VEH_DETECTION_DIR = PROJECT_ROOT.parent.parent.parent
VIDEO_PATH = PED_VEH_DETECTION_DIR / "02_clips" / "01_site1" / "01_individual" / "1.mp4"

K = {
    1: (122, 285), 2: (518, 260), 3: (998, 255), 4: (1212, 268),
    5: (1257, 498), 6: (1039, 518), 7: (101, 537)
}

ABOVE_WAITING_POLY = np.array([(1003, 253), (1211, 264), (1199, 206), (993, 189), (1004, 249)], dtype=np.int32)
BELOW_WAITING_POLY = np.array([(1035, 521), (1274, 492), (1274, 560), (1048, 602), (1036, 525)], dtype=np.int32)
DETECTION_POLY = np.array([K[i] for i in [1, 2, 3, 4, 5, 6, 7]], dtype=np.int32)
K_AOI_FILTER_POLY = np.array([(54, 579), (55, 232), (1023, 177), (1219, 202), (1273, 509), (1233, 567), (1048, 602)], dtype=np.int32)

TOP_LEFT, TOP_RIGHT = K[1], K[4]
BOTTOM_LEFT, BOTTOM_RIGHT = K[7], K[5]
TOP_Y, BOTTOM_Y = TOP_LEFT[1], BOTTOM_LEFT[1]

def interp(a, b, alpha):
    return (int(a[0] + alpha * (b[0] - a[0])), int(a[1] + alpha * (b[1] - a[1])))

def _is_in_poly(poly, cx, cy):
    return cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) > 0

def _get_short_status_code(overall_status, zone=None):
    if overall_status == "Crossed": return "X"
    if overall_status == "Waiting" and zone is not None:
        if _is_in_poly(ABOVE_WAITING_POLY, *zone): return "WU"
        if _is_in_poly(BELOW_WAITING_POLY, *zone): return "WB"
        return "Waiting"
    elif "Crossing," in overall_status:
        return overall_status.split(", ")[1]
    return "N/A"

def sort_ped_statuses(status_list, direction="Down"):
    order_map = {"WB": 1, "R": 2, "M": 3, "L": 4, "X": 5} if direction == "Up" else {"WU": 1, "L": 2, "M": 3, "R": 4, "X": 5}
    return sorted(list(set(status_list)), key=lambda s: order_map.get(s, 99))


class PedestrianAnalyzer:
    def __init__(self, video_path):
        self.video_path = str(video_path)
        self.video_name = Path(video_path).stem

        if not Path(MODEL_PATH).exists():
            raise FileNotFoundError(f"Missing weights file at: {MODEL_PATH}")

        self.model = YOLO(MODEL_PATH)
        self.device = 'mps' if subprocess.run(["sysctl", "hw.optional.arm64"], capture_output=True).returncode == 0 else 'cpu'
        self.tracker = sv.ByteTrack()

        self.detection_polygon = DETECTION_POLY
        self.lane_lines = self._compute_lane_lines()
        self.lane_coefficients = self._precompute_lane_equations()
        self.grid = self._compute_grid(num_meters=45)
        self.aoi_polygon_coords = K_AOI_FILTER_POLY
        self.aoi_zone = sv.PolygonZone(polygon=self.aoi_polygon_coords, triggering_anchors=[sv.Position.CENTER])

        # Tracking Storage Maps
        self.ped_history, self.ped_total_waiting_time, self.ped_frozen_waiting_time = {}, {}, {}
        self.is_crossing_started, self.last_pos, self.ped_crossed_zones, self.ped_overall_status = {}, {}, {}, {}
        self.veh_speed_history, self.veh_segment_cross_time, self.veh_last_m_band_crossed = {}, {}, {}
        self.veh_first_seen, self.last_meter_band, self.last_meter_time, self.veh_current_lane = {}, {}, {}, {}
        self.veh_time_at_ref_line, self.live_veh_gaps = {}, {}
        self.last_global_ref_time = None
        self.pair_max_tta, self.pair_crossing_time, self.pair_min_dist, self.pair_status_history = {}, {}, {}, {}
        self.finalized_interactions = []
        self.fps = 30.0

    def _bbox_center(self, xyxy):
        return int((xyxy[0] + xyxy[2]) // 2), int((xyxy[1] + xyxy[3]) // 2)

    def _compute_lane_lines(self):
        TL, BL = np.array(TOP_LEFT), np.array(BOTTOM_LEFT)
        TR, BR = np.array(TOP_RIGHT), np.array(BOTTOM_RIGHT)
        return {
            "L1": (tuple((TL + 0.3333 * (BL - TL)).astype(int)), tuple((TR + 0.3333 * (BR - TR)).astype(int))),
            "L2": (tuple((TL + 0.6666 * (BL - TL)).astype(int)), tuple((TR + 0.6666 * (BR - TR)).astype(int)))
        }

    def _precompute_lane_equations(self):
        coefs = {}
        for lane_id in ["L1", "L2"]:
            ptA, ptB = self.lane_lines[lane_id]
            if ptB[0] - ptA[0] != 0:
                m = (ptB[1] - ptA[1]) / (ptB[0] - ptA[0])
                coefs[lane_id] = (m, ptA[1] - m * ptA[0], False)
            else:
                coefs[lane_id] = (ptA[0], 0, True)
        return coefs

    def _evaluate_line_y(self, lane_id, cx):
        m, c, is_vertical = self.lane_coefficients[lane_id]
        return m if is_vertical else m * cx + c

    def _get_veh_lane_only(self, cx, cy):
        if cv2.pointPolygonTest(self.detection_polygon, (float(cx), float(cy)), False) < 0: return "Approach"
        return "L" if cy < self._evaluate_line_y("L1", cx) else ("M" if cy < self._evaluate_line_y("L2", cx) else "R")

    def _get_zone(self, pid, cx, cy, t):
        if cv2.pointPolygonTest(self.detection_polygon, (float(cx), float(cy)), False) < 0:
            return "Above" if cy < TOP_Y else "Below"
        return "L" if cy < self._evaluate_line_y("L1", cx) else ("M" if cy < self._evaluate_line_y("L2", cx) else "R")

    def _compute_grid(self, num_meters=45):
        lines = []
        for m in range(num_meters + 1):
            a = 1 - m / num_meters
            lines.append({"m": m, "top": interp(TOP_LEFT, TOP_RIGHT, a), "bottom": interp(BOTTOM_LEFT, BOTTOM_RIGHT, a)})
        return lines

    def _process_pipeline(self, frame):
        results = self.model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False, device=self.device, imgsz=YOLO_INTERNAL_RESOLUTION)[0]
        all_detections = sv.Detections.from_ultralytics(results)
        
        filtered_detections = all_detections[self.aoi_zone.trigger(all_detections)]
        if len(filtered_detections) == 0:
            return sv.Detections.empty(), sv.Detections.empty()

        tracked_detections = self.tracker.update_with_detections(filtered_detections)
        if len(tracked_detections) == 0:
            return sv.Detections.empty(), sv.Detections.empty()

        veh_mask = (tracked_detections.class_id == 0)
        ped_mask = (tracked_detections.class_id == 1) | (tracked_detections.class_id == 2)

        peds = tracked_detections[ped_mask]
        vehs = tracked_detections[veh_mask]

        if len(peds) > 0:
            boxes = peds.xyxy
            crop_mask = (boxes[:, 0] >= CROP_X_MIN) & (boxes[:, 1] >= CROP_Y_MIN) & \
                        (boxes[:, 2] <= CROP_X_MAX) & (boxes[:, 3] <= CROP_Y_MAX)
            peds = peds[crop_mask]

        return peds, vehs

    def _get_veh_geometry_meters(self, xyxy):
        front_grid = min(self.grid, key=lambda g: abs(g["top"][0] - xyxy[2]))["m"]
        return front_grid

    def _compute_multi_ped_gaps(self, vehs, current_peds_map, timestamp):
        for v_idx in range(len(vehs)):
            vid = vehs.tracker_id[v_idx]
            xyxy = vehs.xyxy[v_idx]

            front_m = self._get_veh_geometry_meters(xyxy)
            speed = self.veh_speed_history.get(vid, [])[-1] if self.veh_speed_history.get(vid) else 0.0

            for pid, (ped_m, ped_status, ped_pos) in current_peds_map.items():
                pair_key = (vid, pid)
                short_status = _get_short_status_code(ped_status, ped_pos)
                self.pair_status_history.setdefault(pair_key, set()).add(short_status)
                dist = front_m - ped_m

                if speed > 0.5:
                    tta = dist / speed
                    if front_m > ped_m:
                        if tta > self.pair_max_tta.get(pair_key, -float('inf')):
                            self.pair_max_tta[pair_key] = tta
                            self.pair_min_dist[pair_key] = dist
                elif speed <= 0.5 and front_m > ped_m:
                    if pair_key not in self.pair_max_tta:
                        self.pair_max_tta[pair_key] = 0.0
                        self.pair_min_dist[pair_key] = dist

                if front_m <= ped_m and pair_key in self.pair_max_tta and pair_key not in self.pair_crossing_time:
                    self.pair_crossing_time[pair_key] = timestamp
                    self._log_interaction(vid, pid, timestamp)

    def _log_interaction(self, vid, pid, t1):
        self.finalized_interactions.append({
            "vid": vid, "pid": pid, "t1": t1,
            "d_gap_at_tta": self.pair_min_dist.get((vid, pid), 0.0),
            "lane_t1": self.veh_current_lane.get(vid, 'N/A')
        })

    def _track_segment_speed(self, vehs, timestamp):
        for i in range(len(vehs)):
            vid = vehs.tracker_id[i]
            x1, y1, x2, y2 = map(int, vehs.xyxy[i])
            front_m_band = min(self.grid, key=lambda g: abs(g["top"][0] - x2))["m"]
            prev_m_band = self.veh_last_m_band_crossed.get(vid, front_m_band)

            crossed_ref = False
            if prev_m_band > HEADWAY_REFERENCE_METER and front_m_band <= HEADWAY_REFERENCE_METER:
                crossed_ref = True
            elif front_m_band <= HEADWAY_REFERENCE_METER and vid not in self.veh_time_at_ref_line:
                crossed_ref = True

            if crossed_ref:
                self.veh_time_at_ref_line[vid] = timestamp
                if self.last_global_ref_time is not None:
                    self.live_veh_gaps[vid] = timestamp - self.last_global_ref_time
                else:
                    self.live_veh_gaps[vid] = 0.0
                self.last_global_ref_time = timestamp

            if front_m_band != prev_m_band and front_m_band < prev_m_band:
                prev_line_x = [g['top'][0] for g in self.grid if g['m'] == prev_m_band][0]
                if x2 > prev_line_x:
                    if (vid, prev_m_band) not in self.veh_segment_cross_time:
                        self.veh_segment_cross_time[(vid, prev_m_band)] = timestamp
                    self.veh_last_m_band_crossed[vid] = front_m_band

            if vid not in self.veh_last_m_band_crossed:
                self.veh_last_m_band_crossed[vid] = front_m_band
                self.veh_segment_cross_time[(vid, front_m_band)] = timestamp

    def _calculate_full_ped_metrics(self, pid):
        df = pd.DataFrame(self.ped_history.get(pid, []), columns=["t", "cx", "cy", "zone"])
        if df.empty or not df[df.zone.isin(["L", "M", "R"])].shape[0] >= 1: return None

        past_lanes = {z for z in df["zone"] if z in ["L", "M", "R"]}
        if len(past_lanes) < 2: return None

        waiting = self.ped_frozen_waiting_time.get(pid, 0.0)
        lane_entries = df.loc[df.zone.isin(["L", "M", "R"]), "t"]
        cross_start = lane_entries.min()
        cross_end = lane_entries.max()

        l_entry = df.loc[df.zone == "L", "t"].min()
        m_entry = df.loc[df.zone == "M", "t"].min()
        r_entry = df.loc[df.zone == "R", "t"].min()

        direction = "Down"
        valid_entries = []
        if pd.notna(l_entry): valid_entries.append((l_entry, "L"))
        if pd.notna(m_entry): valid_entries.append((m_entry, "M"))
        if pd.notna(r_entry): valid_entries.append((r_entry, "R"))
        valid_entries.sort(key=lambda x: x[0])

        if len(valid_entries) >= 2:
            first, last = valid_entries[0][1], valid_entries[-1][1]
            if (first in ["R", "M"]) and last == "L": direction = "Up"
            elif first == "R" and last == "M": direction = "Up"

        t_exit_L, t_exit_M, t_exit_R = None, None, None
        timeline = []
        if direction == "Down":
            t_exit_L = m_entry if pd.notna(m_entry) else (r_entry if pd.notna(r_entry) else cross_end)
            t_exit_M = r_entry if pd.notna(r_entry) else cross_end
            t_exit_R = cross_end
            if pd.notna(l_entry): timeline.append(("L", l_entry, t_exit_L))
            if pd.notna(m_entry): timeline.append(("M", m_entry, t_exit_M))
            if pd.notna(r_entry): timeline.append(("R", r_entry, t_exit_R))
        else:
            t_exit_R = m_entry if pd.notna(m_entry) else (l_entry if pd.notna(l_entry) else cross_end)
            t_exit_M = l_entry if pd.notna(l_entry) else cross_end
            t_exit_L = cross_end
            if pd.notna(r_entry): timeline.append(("R", r_entry, t_exit_R))
            if pd.notna(m_entry): timeline.append(("M", m_entry, t_exit_M))
            if pd.notna(l_entry): timeline.append(("L", l_entry, t_exit_L))

        # UPGRADE JITTER PROTECTION: Prevents extremely high outlier split speeds near dividing lane borders
        def calc_speed(entry, exit_t):
            if pd.isna(entry) or pd.isna(exit_t): return None
            duration = exit_t - entry
            if duration <= 0.001: return None
            raw_speed = LANE_WIDTH / duration
            return min(raw_speed, 4.5) if duration < 0.25 else raw_speed  # Filters out noise anomalies

        # FIX: Swapped implicit evaluation lines for robust structural 'is not None' checking rules
        has_valid_timeline = (cross_start is not None) and (cross_end is not None)
        overall_spd = (3 * LANE_WIDTH / (cross_end - cross_start)) if (has_valid_timeline and (cross_end - cross_start) > 0.001) else None

        return {
            "pid": pid, "direction": direction,
            "T_start_crossing": cross_start, "frozen_wait_time": waiting, "cross_end": cross_end,
            "T_exit_L": t_exit_L, "T_exit_M": t_exit_M, "T_exit_R": t_exit_R,
            "speed_L": calc_speed(l_entry, t_exit_L), "speed_M": calc_speed(m_entry, t_exit_M), "speed_R": calc_speed(r_entry, t_exit_R),
            "overall_speed": overall_spd,
            "timeline": timeline
        }

    def generate_pedestrian_reports(self):
        reports = {}
        unique_pids = sorted(list(set(d['pid'] for d in self.finalized_interactions)))
        speed_map = self.compute_vehicle_speed_metrics().set_index('ID')['Average_Speed'].to_dict()

        for target_pid in unique_pids:
            ped_v = self._calculate_full_ped_metrics(target_pid)
            if not ped_v: 
                continue

            rows = []
            ped_dir = ped_v.get("direction", "Down")
            finish_time = ped_v.get("T_exit_R" if ped_dir == "Down" else "T_exit_L") or ped_v.get("cross_end", 99999.0)

            ped_interactions = sorted([d for d in self.finalized_interactions if d['pid'] == target_pid], key=lambda x: x['t1'])
            filtered_interactions = [d for d in ped_interactions if not (isinstance(finish_time, (int, float)) and self.veh_first_seen.get(d["vid"], 0.0) > finish_time)]

            valid_rows_data = []
            has_accepted_gap = False

            for d in filtered_interactions:
                vid = d["vid"]
                veh_t0 = self.veh_first_seen.get(vid, 0.0)
                veh_t1 = d["t1"]
                d_gap_display = d["d_gap_at_tta"]

                real_status = set()
                start_cross = ped_v.get("T_start_crossing")
                if isinstance(start_cross, (int, float)) and veh_t0 < start_cross: real_status.add("WU" if ped_dir == "Down" else "WB")
                for lane, l_start, l_end in ped_v.get("timeline", []):
                    if max(veh_t0, l_start) < min(veh_t1, l_end): real_status.add(lane)
                if isinstance(ped_v.get("cross_end"), (int, float)) and veh_t1 > ped_v.get("cross_end"): real_status.add("X")
                
                ped_status_display = ", ".join(sort_ped_statuses(list(real_status), direction=ped_dir))
                decision = "Rejected"
                
                if bool(real_status.intersection({"L", "M", "R", "X"})) and d["lane_t1"] in ["L", "M", "R"]:
                    lane_exit_time = ped_v.get(f"T_exit_{d['lane_t1']}")
                    if isinstance(veh_t1, (int, float)) and isinstance(lane_exit_time, (int, float)) and lane_exit_time < veh_t1:
                        decision = "Accepted"

                if isinstance(finish_time, (int, float)) and isinstance(veh_t1, (int, float)) and veh_t1 > finish_time:
                    d_gap_display = 36.00
                    decision = "Accepted"

                row_data = {
                    "Video_Name": self.video_name, "Veh_ID": vid, "Ped_ID": target_pid, "Veh_T1_Time_Val": d['t1'], "Veh_T1_Time": f"{d['t1']:.2f}",
                    "Ped_Start_Time": f"{ped_v.get('T_start_crossing', 'N/A'):.2f}" if isinstance(ped_v.get('T_start_crossing'), (int, float)) else "N/A",
                    "Ped_Direction": ped_dir, "Ped_Decision": decision, "Veh_Dist_Gap": f"{d_gap_display:.2f}" if isinstance(d_gap_display, (int, float)) else d_gap_display,
                    "Veh_Lane": d["lane_t1"], "Ped_Status": ped_status_display, "Veh_Speed": f"{speed_map.get(vid, 0.0):.2f}" if isinstance(speed_map.get(vid), (int, float)) else "N/A",
                    "Ped_Wait_Time": f"{ped_v.get('frozen_wait_time', 0.0):.2f}" if isinstance(ped_v.get('frozen_wait_time'), (int, float)) else "N/A",
                    "Ped_Avg_Speed": f"{ped_v.get('overall_speed', 0.0):.2f}" if isinstance(ped_v.get('overall_speed'), (int, float)) else "N/A",
                    "Ped_Speed_L": f"{ped_v.get('speed_L', 0.0):.2f}" if isinstance(ped_v.get('speed_L'), (int, float)) else "N/A",
                    "Ped_Speed_M": f"{ped_v.get('speed_M', 0.0):.2f}" if isinstance(ped_v.get('speed_M'), (int, float)) else "N/A",
                    "Ped_Speed_R": f"{ped_v.get('speed_R', 0.0):.2f}" if isinstance(ped_v.get('speed_R'), (int, float)) else "N/A",
                }

                if decision == "Rejected":
                    valid_rows_data.append(row_data)
                elif decision == "Accepted" and not has_accepted_gap:
                    valid_rows_data.append(row_data)
                    has_accepted_gap = True

            valid_rows_data.sort(key=lambda x: x["Veh_T1_Time_Val"])
            previous_t1 = None
            for r in valid_rows_data:
                current_t1 = r["Veh_T1_Time_Val"]
                r["Veh_Headway"] = f"{current_t1 - previous_t1:.2f}" if previous_t1 is not None else f"{self.live_veh_gaps.get(r['Veh_ID'], 0.0):.2f}"
                previous_t1 = current_t1
                del r["Veh_T1_Time_Val"]
                rows.append(r)

            if rows: reports[target_pid] = pd.DataFrame(rows)

        return reports

    def compute_vehicle_speed_metrics(self):
        rows = []
        for vid in {d["vid"] for d in self.finalized_interactions}:
            speeds = self.veh_speed_history.get(vid, [])
            rows.append({"ID": vid, "Average_Speed": np.mean(speeds) if speeds else 'N/A'})
        return pd.DataFrame(rows)

    def _draw(self, frame, peds, vehs, timestamp):
        overlay = frame.copy()
        cv2.polylines(frame, [self.detection_polygon], True, (255, 0, 0), 2)
        cv2.polylines(frame, [self.aoi_polygon_coords], True, (255, 255, 0), 1)

        for name, line_pts in self.lane_lines.items():
            cv2.line(frame, line_pts[0], line_pts[1], (255, 0, 150), 2, lineType=cv2.LINE_AA)
            cv2.putText(frame, name, (line_pts[0][0] + 10, line_pts[0][1] - 10), 0, 0.4, (255, 0, 150), 1)

        cv2.fillPoly(overlay, [ABOVE_WAITING_POLY], (0, 150, 255))
        cv2.fillPoly(overlay, [BELOW_WAITING_POLY], (255, 150, 0))
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        current_peds_map = {}
        for i in range(len(peds)):
            pid = peds.tracker_id[i]
            cid = peds.class_id[i]  
            x1, y1, x2, y2 = map(int, peds.xyxy[i])
            cx, cy = self._bbox_center(peds.xyxy[i])
            zone = self._get_zone(pid, cx, cy, timestamp)
            self.ped_history.setdefault(pid, []).append((timestamp, cx, cy, zone))

            if pid not in self.last_pos:
                self.last_pos[pid] = (cx, cy, timestamp)
                self.ped_total_waiting_time[pid] = 0.0
                self.is_crossing_started[pid] = False
            else:
                if zone in ["L", "M", "R"] and not self.is_crossing_started[pid]:
                    self.is_crossing_started[pid] = True
                    self.ped_frozen_waiting_time[pid] = self.ped_total_waiting_time.get(pid, 0.0)
                if zone in ["Above", "Below"] and not self.is_crossing_started[pid]:
                    self.ped_total_waiting_time[pid] += (timestamp - self.last_pos[pid][2])
                self.last_pos[pid] = (cx, cy, timestamp)

            if zone in ["L", "M", "R"]:
                status = f"Crossing ({zone})"
            else:
                if self.is_crossing_started[pid]:
                    past_zones = {h[3] for h in self.ped_history[pid] if h[3] in ["L", "M", "R"]}
                    if len(past_zones) >= 2:
                        status = "Crossed"
                    else:
                        status = f"Waiting ({zone})"
                else:
                    status = f"Waiting ({zone})"

            self.ped_overall_status[pid] = status
            m_pos = min(self.grid, key=lambda g: abs(g["top"][0] - cx))["m"]
            current_peds_map[pid] = (m_pos, status, (cx, cy))

            if cid == 2:
                box_color = (255, 0, 255) 
                label_tag = f"G{pid}: {status}"
            else:
                box_color = (0, 0, 255) if "Crossing" in status else (0, 255, 0)
                label_tag = f"P{pid}: {status}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, label_tag, (x1, y1 - 5), 0, 0.5, (255, 255, 255), 2)

        self._compute_multi_ped_gaps(vehs, current_peds_map, timestamp)
        self._track_segment_speed(vehs, timestamp)

        for i in range(len(vehs)):
            vid = vehs.tracker_id[i]
            x1, y1, x2, y2 = map(int, vehs.xyxy[i])
            cx, cy = self._bbox_center(vehs.xyxy[i])
            lane_str = self._get_veh_lane_only(cx, cy)
            self.veh_current_lane[vid] = lane_str

            curr_m = min(self.grid, key=lambda g: abs(g["top"][0] - cx))["m"]
            if vid not in self.veh_first_seen:
                self.veh_first_seen[vid] = timestamp
                self.last_meter_band[vid] = curr_m
                self.last_meter_time[vid] = timestamp

            prev_m, prev_t = self.last_meter_band.get(vid), self.last_meter_time.get(vid)
            if prev_m is not None and (timestamp - prev_t) > 0 and curr_m != prev_m:
                speed = abs(curr_m - prev_m) / (timestamp - prev_t)
                if speed > 0: self.veh_speed_history.setdefault(vid, []).append(speed)
                self.last_meter_band[vid] = curr_m
                self.last_meter_time[vid] = timestamp

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 150, 0), 2)
            cv2.putText(frame, f"V{vid} {lane_str}", (x1, y1 - 5), 0, 0.5, (255, 150, 0), 2)

        return frame

    def process_video(self):
        if not Path(self.video_path).exists():
            raise FileNotFoundError(f"Missing source file: {self.video_path}")
            
        cap = cv2.VideoCapture(str(self.video_path))
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_i = 0

        print(f"[INFO] Running fast pipeline on {self.device.upper()} core layout...")
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            ts = frame_i / self.fps
            
            p, v = self._process_pipeline(frame)
            
            if frame_i % FRAME_SKIP == 0:
                frame = self._draw(frame, p, v, ts)
                cv2.imshow("Multi-Ped Analytics", frame)
                if cv2.waitKey(1) == 27: break
                
            frame_i += 1

        cap.release()
        cv2.destroyAllWindows()

        ped_reports = self.generate_pedestrian_reports()
        all_dfs = []
        for pid, df in ped_reports.items():
            print(f"\n========== METRICS FOR PEDESTRIAN P{pid} ==========")
            print(df.to_markdown(index=False))
            all_dfs.append(df)

        if all_dfs:
            final_df = pd.concat(all_dfs, ignore_index=True)
            output_dir = PROJECT_ROOT / "z_output"
            output_dir.mkdir(exist_ok=True)
            csv_path = output_dir / f"{self.video_name}.csv"
            final_df.to_csv(csv_path, index=False)
            print(f"\n[INFO] Analytics complete. Report saved: {csv_path}")


if __name__ == "__main__":
    PedestrianAnalyzer(VIDEO_PATH).process_video()