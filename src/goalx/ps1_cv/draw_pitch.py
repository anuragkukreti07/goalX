import cv2
import numpy as np

# CONFIG: Scale (Pixels per Meter)
# 1 meter = 8 pixels. 
# So a 105m pitch will be 840 pixels wide.
SCALE = 8  
PITCH_LENGTH = 105
PITCH_WIDTH = 68

def draw_pitch(show=True):
    # 1. Create a green canvas
    # Dimensions: (Height, Width)
    h = PITCH_WIDTH * SCALE
    w = PITCH_LENGTH * SCALE
    
    # Standard "Pitch Green" color (BGR)
    img = np.ones((h, w, 3), dtype=np.uint8) * np.array([50, 160, 50], dtype=np.uint8)

    # 2. Define colors & thickness
    WHITE = (255, 255, 255)
    THICKNESS = 2

    # 3. Draw Outer Boundary
    cv2.rectangle(img, (0, 0), (w, h), WHITE, THICKNESS)

    # 4. Draw Center Line
    mid_x = w // 2
    cv2.line(img, (mid_x, 0), (mid_x, h), WHITE, THICKNESS)

    # 5. Draw Center Circle
    center_radius = int(9.15 * SCALE)  # 9.15m is standard radius
    cv2.circle(img, (mid_x, h // 2), center_radius, WHITE, THICKNESS)
    # Center Spot
    cv2.circle(img, (mid_x, h // 2), 4, WHITE, -1)

    # 6. Draw Penalty Areas (Left & Right)
    # Dimensions: 16.5m deep, 40.3m wide (approx)
    pen_depth = int(16.5 * SCALE)
    pen_width = int(40.32 * SCALE)
    pen_y_start = (h - pen_width) // 2
    
    # Left Penalty Box
    cv2.rectangle(img, (0, pen_y_start), (pen_depth, pen_y_start + pen_width), WHITE, THICKNESS)
    # Right Penalty Box
    cv2.rectangle(img, (w - pen_depth, pen_y_start), (w, pen_y_start + pen_width), WHITE, THICKNESS)

    # 7. Draw Goal Areas (6-yard box)
    # Dimensions: 5.5m deep, 18.32m wide
    goal_depth = int(5.5 * SCALE)
    goal_width = int(18.32 * SCALE)
    goal_y_start = (h - goal_width) // 2

    # Left Goal Box
    cv2.rectangle(img, (0, goal_y_start), (goal_depth, goal_y_start + goal_width), WHITE, THICKNESS)
    # Right Goal Box
    cv2.rectangle(img, (w - goal_depth, goal_y_start), (w, goal_y_start + goal_width), WHITE, THICKNESS)

    # Save
    cv2.imwrite("data/pitch_map.png", img)
    print("✅ Generated 2D Pitch: data/pitch_map.png")
    
    if show:
        cv2.imshow("Tactical Board", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    draw_pitch()