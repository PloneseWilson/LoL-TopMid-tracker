import cv2
import numpy as np
import time
from dataclasses import dataclass
from typing import Literal


@dataclass
class Circle:
    x: int
    y: int
    r: int
    color: Literal["blue", "red"]


# ── Lane zones (calibrated from 265x268 minimap) ─────────────────────────────

# Top lane: radius from top-right corner (img_w, 0)  — red line length = 287px
# Bot lane: same radius from bottom-left corner (0, img_h)
# Mid lane: circle centered at (128,128) radius 39px

LANE_RADIUS     = 130
MID_LANE_CENTER = (128, 128)
MID_LANE_RADIUS = 50

# Image dimensions the zones were calibrated on
_CAL_W, _CAL_H = 265, 268


def _scale_radius(r: int, img_w: int, img_h: int) -> float:
    """Scale calibrated radius to actual image size."""
    scale = np.hypot(img_w, img_h) / np.hypot(_CAL_W, _CAL_H)
    return r * scale


def in_top_lane(x: int, y: int, img_w: int = _CAL_W, img_h: int = _CAL_H) -> bool:
    """Point is within LANE_RADIUS of the top-left corner."""
    r = _scale_radius(LANE_RADIUS, img_w, img_h)
    return np.hypot(x, y) <= r



def in_mid_lane(x: int, y: int) -> bool:
    return np.hypot(x - MID_LANE_CENTER[0], y - MID_LANE_CENTER[1]) <= MID_LANE_RADIUS


def classify_lane(c: Circle, img_w: int = _CAL_W, img_h: int = _CAL_H) -> Literal["top", "mid", "bot", "other"]:
    if in_mid_lane(c.x, c.y):
        return "mid"
    if in_top_lane(c.x, c.y, img_w, img_h):
        return "top"
    return "other"


# ── Highland filter ───────────────────────────────────────────────────────────

def is_in_highland(x: int, y: int, img_w: int, img_h: int, threshold: int = 115) -> bool:
    dist_bottom_left = np.hypot(x,         y - img_h)
    dist_top_right   = np.hypot(x - img_w, y)
    return dist_bottom_left <= threshold or dist_top_right <= threshold


# ── Detection ─────────────────────────────────────────────────────────────────

def detect_circles(img: np.ndarray, color: Literal["blue", "red"]) -> list[Circle]:
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    if color == "blue":
        mask = cv2.inRange(hsv, np.array([95, 40, 60]), np.array([135, 255, 255]))
    else:
        mask1 = cv2.inRange(hsv, np.array([0,   40, 60]), np.array([5,  255, 255]))
        mask2 = cv2.inRange(hsv, np.array([175, 40, 60]), np.array([180, 255, 255]))
        mask  = mask1 | mask2

    raw = cv2.HoughCircles(
        mask, cv2.HOUGH_GRADIENT,
        dp=1, minDist=15, param1=30, param2=8,
        minRadius=10, maxRadius=15
    )

    if raw is None:
        return []

    return [
        Circle(x=int(x), y=int(y), r=int(r), color=color)
        for x, y, r in np.round(raw[0]).astype(int)
        if not is_in_highland(int(x), int(y), w, h)
    ]


def detect_and_annotate(img: np.ndarray) -> tuple[list[Circle], np.ndarray]:
    blue = detect_circles(img, "blue")
    red  = detect_circles(img, "red")
    circles = blue + red
    annotated = annotate_image(img, circles)
    return circles, annotated


# ── Annotation ────────────────────────────────────────────────────────────────

def annotate_image(img: np.ndarray, circles: list[Circle]) -> np.ndarray:
    COLOR = {"blue": (255, 0, 0), "red": (0, 0, 255)}
    out = img.copy()
    label_counts = {"blue": 0, "red": 0}
    for c in circles:
        label_counts[c.color] += 1
        color = COLOR[c.color]
        cv2.circle(out, (c.x, c.y), c.r, color, 2)
        cv2.circle(out, (c.x, c.y), 2,   color, -1)
        cv2.putText(out, f"{c.color[0].upper()}{label_counts[c.color]}", (c.x + c.r + 2, c.y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return out


# ── Lane alert logic ──────────────────────────────────────────────────────────

def build_lane_alerts(
    circles: list[Circle],
    enemy_color: Literal["blue", "red"],
) -> list[str]:
    """
    Given all detected circles and the enemy color, return alert strings per lane.

    Rules:
      - Only count enemy_color circles for lane presence
      - Top lane:  1 enemy  → "Top lane: enemy present"
                   2+       → "Top lane: multiple players"
      - Bot lane:  same rules
      - Mid lane:  0 enemy  → "Mid lane: missing"
                   1+       → "Mid lane: enemy present"
    """
    enemies = [c for c in circles if c.color == enemy_color]

    img_w, img_h = _CAL_W, _CAL_H  # default; override if needed
    top = [c for c in enemies if in_top_lane(c.x, c.y, img_w, img_h)]
    mid = [c for c in enemies if in_mid_lane(c.x, c.y)]

    alerts = []

    if len(top) == 0:
        alerts.append("Top: missing")
    elif len(top) == 1:
        alerts.append("Top: exist (1)")
    else:
        alerts.append(f"Top: multiple ({len(top)})")

    if len(mid) == 0:
        alerts.append("Mid: missing")
    elif len(mid) == 1:
        alerts.append("Mid: exist (1)")
    else:
        alerts.append(f"Mid: multiple ({len(mid)})")

    return alerts


# ── Position window ───────────────────────────────────────────────────────────

def render_position_window(
    alerts: list[str],
) -> np.ndarray:
    panel = np.full((110, 250, 3), (28, 32, 36), dtype=np.uint8)

    for i, alert in enumerate(alerts):
        color = (0, 200, 255) if "missing" in alert.lower() else (100, 220, 100)
        cv2.putText(panel, alert, (10, 40 + i * 46), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return panel


# ── Screen capture ────────────────────────────────────────────────────────────

def capture_screen_region(left: int, top: int, width: int, height: int) -> np.ndarray:
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    gdi32  = ctypes.windll.gdi32
    user32.SetProcessDPIAware()

    screen_dc = user32.GetDC(0)
    memory_dc = gdi32.CreateCompatibleDC(screen_dc)
    bitmap    = gdi32.CreateCompatibleBitmap(screen_dc, width, height)
    old_bitmap = gdi32.SelectObject(memory_dc, bitmap)

    try:
        gdi32.BitBlt(memory_dc, 0, 0, width, height, screen_dc, left, top, 0x00CC0020)

        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [("biSize",wintypes.DWORD),("biWidth",wintypes.LONG),
                        ("biHeight",wintypes.LONG),("biPlanes",wintypes.WORD),
                        ("biBitCount",wintypes.WORD),("biCompression",wintypes.DWORD),
                        ("biSizeImage",wintypes.DWORD),("biXPelsPerMeter",wintypes.LONG),
                        ("biYPelsPerMeter",wintypes.LONG),("biClrUsed",wintypes.DWORD),
                        ("biClrImportant",wintypes.DWORD)]

        class BITMAPINFO(ctypes.Structure):
            _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 3)]

        bmi = BITMAPINFO()
        bmi.bmiHeader.biSize        = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth       = width
        bmi.bmiHeader.biHeight      = -height
        bmi.bmiHeader.biPlanes      = 1
        bmi.bmiHeader.biBitCount    = 32
        bmi.bmiHeader.biCompression = 0

        buf = ctypes.create_string_buffer(width * height * 4)
        gdi32.GetDIBits(memory_dc, bitmap, 0, height, buf, ctypes.byref(bmi), 0)
        bgra = np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 4))
        return bgra[:, :, :3].copy()
    finally:
        gdi32.SelectObject(memory_dc, old_bitmap)
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(memory_dc)
        user32.ReleaseDC(0, screen_dc)


def get_virtual_screen_bounds() -> tuple[int, int, int, int]:
    import ctypes
    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    return (user32.GetSystemMetrics(76), user32.GetSystemMetrics(77),
            user32.GetSystemMetrics(78), user32.GetSystemMetrics(79))


# ── Live loop ─────────────────────────────────────────────────────────────────

def choose_player_team() -> Literal["blue", "red"]:
    while True:
        choice = input("Your team (blue/red): ").strip().lower()
        if choice in ("b", "blue"):  return "blue"
        if choice in ("r", "red"):   return "red"
        print("Please type blue or red.")


def run_live_minimap(
    crop_width:  int   = 278,
    crop_height: int   = 279,
    offset_x:    int   = 0,
    offset_y:    int   = 0,
    interval:    float = 1.0,
    zoom:        float = 1.5,
    player_team: Literal["blue", "red"] | None = None,
) -> None:
    screen_left, screen_top, screen_w, screen_h = get_virtual_screen_bounds()
    left = screen_left + screen_w - crop_width  - offset_x
    top  = screen_top  + screen_h - crop_height - offset_y
    enemy_color: Literal["blue", "red"] | None = None
    if player_team is not None:
        enemy_color = "red" if player_team == "blue" else "blue"

    print(f"Capturing bottom-right  left={left} top={top}  {crop_width}x{crop_height}")
    print(f"Enemy color: {enemy_color or 'both'}   Press 0 to stop.")

    minimap_win   = "LoL minimap labels"
    positions_win = "LoL minimap positions"
    cv2.namedWindow(minimap_win,   cv2.WINDOW_NORMAL)
    cv2.namedWindow(positions_win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(minimap_win,   int(crop_width * zoom), int(crop_height * zoom))
    cv2.resizeWindow(positions_win, 250, 110)
    cv2.moveWindow(minimap_win,    40, 40)
    cv2.moveWindow(positions_win,  40, 80 + int(crop_height * zoom))
    for win in (minimap_win, positions_win):
        try: cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)
        except cv2.error: pass

    while True:
        start   = time.monotonic()
        minimap = capture_screen_region(left, top, crop_width, crop_height)

        circles, _ = detect_and_annotate(minimap)
        shown      = [c for c in circles if enemy_color is None or c.color == enemy_color]
        annotated  = annotate_image(minimap, shown)
        alerts     = build_lane_alerts(circles, enemy_color or "red")

        checked_at = time.strftime("%H:%M:%S")
        status = f"{len(shown)} {enemy_color or 'all'} | {checked_at}"
        cv2.putText(annotated, status, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 2)
        cv2.putText(annotated, status, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0),       1)

        positions = render_position_window(alerts)

        cv2.imshow(minimap_win,   annotated)
        cv2.imshow(positions_win, positions)

        elapsed = time.monotonic() - start
        key = cv2.waitKey(max(1, int((interval - elapsed) * 1000)))
        if key == 48:  # 0
            break

    cv2.destroyWindow(minimap_win)
    cv2.destroyWindow(positions_win)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path",          nargs="?", default=None)
    parser.add_argument("--live",        action="store_true")
    parser.add_argument("--crop-width",  type=int,   default=278)
    parser.add_argument("--crop-height", type=int,   default=279)
    parser.add_argument("--offset-x",   type=int,   default=0)
    parser.add_argument("--offset-y",   type=int,   default=0)
    parser.add_argument("--interval",   type=float, default=1.0)
    parser.add_argument("--zoom",        type=float, default=1.5)
    parser.add_argument("--team",        choices=("blue","red"), default=None)
    args = parser.parse_args()

    if args.live or args.path is None:
        player_team = args.team if args.team else choose_player_team()
        run_live_minimap(
            crop_width=args.crop_width, crop_height=args.crop_height,
            offset_x=args.offset_x,    offset_y=args.offset_y,
            interval=args.interval,     zoom=args.zoom,
            player_team=player_team,
        )
        raise SystemExit(0)

    img = cv2.imread(args.path)
    if img is None:
        raise FileNotFoundError(f"Could not read: {args.path}")

    circles, annotated = detect_and_annotate(img)
    print(f"Detected {len(circles)} players:")
    for c in circles:
        print(f"  [{c.color:4s}] center=({c.x}, {c.y})  lane={classify_lane(c)}")

    cv2.imwrite(args.path.replace(".png", "_annotated.png"), annotated)