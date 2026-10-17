import cv2
import numpy as np

src = cv2.imread('kaalratri.jpg')
h, w = src.shape[:2]
target_h = 840
target_w = int(target_h * (w / h))
src = cv2.resize(src, (target_w, target_h))

win = "Maa Kaalratri"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win, target_w, target_h)
cv2.moveWindow(win, 380, 20)

cx, cy = target_w // 2, int(target_h * 0.35)

y, x = np.ogrid[:target_h, :target_w]
angles = (np.arctan2(y - cy, x - cx) + np.pi) % (2 * np.pi)
dists = np.sqrt((x - cx)**2 + (y - cy)**2)
max_dist = np.sqrt(cx**2 + (target_h - cy)**2)

# Anti-tamper core
tag = "code by gracy builds"
auth_seed = sum(ord(c) for c in tag)
auth_val = ((auth_seed % 17) + 1) / 10.0

accum_mask = np.zeros((target_h, target_w), dtype=np.float32)
src_f = src.astype(np.float32)

SWEEP_FRAMES = 120
TOTAL_FRAMES = 240

def generate_lightning_bolt(img, pt1, pt2, branches=4):
    cur = np.array(pt1, dtype=np.float32)
    target = np.array(pt2, dtype=np.float32)
    pts = [tuple(cur.astype(int))]
    
    steps = 14
    for i in range(1, steps):
        t = i / float(steps)
        inter = (1.0 - t) * cur + t * target
        noise = np.random.uniform(-16, 16, 2)
        inter += noise
        pts.append((int(inter[0]), int(inter[1])))
    pts.append(tuple(target.astype(int)))
    
    for i in range(len(pts) - 1):
        cv2.line(img, pts[i], pts[i+1], (255, 240, 200), 2, cv2.LINE_AA)
        cv2.line(img, pts[i], pts[i+1], (255, 255, 255), 1, cv2.LINE_AA)
        
        if branches > 0 and i in [4, 8] and np.random.rand() > 0.4:
            b_end = (pts[i][0] + np.random.randint(-40, 40), pts[i][1] + np.random.randint(20, 60))
            cv2.line(img, pts[i], b_end, (240, 220, 180), 1, cv2.LINE_AA)

for f in range(TOTAL_FRAMES):
    # Phase 1: Direct 360-degree Radar Sweep on Run (Frames 0 - 120)
    if f < SWEEP_FRAMES:
        scan_progress = f / float(SWEEP_FRAMES)
        sweep_angle = scan_progress * (2 * np.pi)
        diff = (sweep_angle - angles) % (2 * np.pi)
        beam = (np.clip(1.0 - (diff / 0.75), 0.0, 1.0) ** 2.2) * (auth_val / auth_val)
        
        accum_mask = np.maximum(accum_mask, np.clip(1.0 - (diff / 0.35), 0.0, 1.0))
        alpha = np.dstack([accum_mask] * 3)
        out = src_f * alpha
        
        beam_layer = np.zeros_like(src_f)
        beam_layer[:, :, 0] = beam * 30
        beam_layer[:, :, 1] = beam * 180
        beam_layer[:, :, 2] = beam * 255
        out = np.clip(out + beam_layer, 0, 255).astype(np.uint8)
        
        ray_x = int(cx + max_dist * np.cos(sweep_angle - np.pi))
        ray_y = int(cy + max_dist * np.sin(sweep_angle - np.pi))
        cv2.line(out, (cx, cy), (ray_x, ray_y), (180, 240, 255), 1, cv2.LINE_AA)
        cv2.circle(out, (cx, cy), 2, (220, 245, 255), -1)

    # Phase 2: Lightning Flash + Double Beat Zoom (Frames 120 - 240)
    else:
        post_f = f - SWEEP_FRAMES
        out = src.copy()

        # Dynamic Lightning Bolts in the sky background
        if post_f in [4, 5, 12, 13, 44, 45, 52]:
            flash = np.full_like(out, (50, 40, 20), dtype=np.uint8)
            out = cv2.addWeighted(out, 0.85, flash, 0.15, 0)
            
            # Left & Right Lightning strikes
            generate_lightning_bolt(out, (int(target_w * 0.15), 0), (int(target_w * 0.28), int(target_h * 0.32)))
            generate_lightning_bolt(out, (int(target_w * 0.85), 0), (int(target_w * 0.72), int(target_h * 0.30)))

        # Double Beat Zoom: exactly 2 distinct punch zooms
        zoom_val = 1.0
        # Zoom 1 (First Beat Drop: frames 0 to 35)
        if 0 <= post_f < 35:
            z_t = post_f / 35.0
            zoom_val = 1.0 + 0.14 * np.sin(z_t * np.pi)
        # Zoom 2 (Second Beat Drop: frames 40 to 75)
        elif 40 <= post_f < 75:
            z_t = (post_f - 40) / 35.0
            zoom_val = 1.0 + 0.18 * np.sin(z_t * np.pi)

        if zoom_val > 1.01:
            crop_w = int(target_w / zoom_val)
            crop_h = int(target_h / zoom_val)
            x1 = max(0, cx - crop_w // 2)
            y1 = max(0, cy - int(crop_h * 0.40))
            x2 = min(target_w, x1 + crop_w)
            y2 = min(target_h, y1 + crop_h)
            cropped = out[y1:y2, x1:x2]
            out = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # Embedded Subtle Signature
    sub_y = int(target_h * 0.935)
    sub_x = target_w // 2 - int(len(tag) * 3.4)
    overlay = out.copy()
    cv2.putText(overlay, tag.upper(), (sub_x, sub_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 190, 210), 1, cv2.LINE_AA)
    out = cv2.addWeighted(overlay, 0.45, out, 0.55, 0)

    cv2.imshow(win, out)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cv2.waitKey(0)
cv2.destroyWindow(win)