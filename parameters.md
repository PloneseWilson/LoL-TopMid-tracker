# End program: press 0  
    1. run_live_minimap: if key == 48: break

# Information Window size: 250 x 110   
    1. run_live_minimap: cv2.resizeWindow(positions_win, 250, 110)
    2. render_position_window: panel = np.full((110, 250, 3), (28, 32, 36), dtype=np.uint8) 

# Minimap capture Size: 278 x 279, any close numbers are fine  
    1. run_live_minimap: crop_width:  int   = 278, crop_height: int   = 279,

# Mid lane zone radius: 50    
    1. MID_LANE_RADIUS = 50

# Top lane zone radius: 130  
    1. LANE_RADIUS     = 130

# Highland neglect zone: 115  
    1. is_in_highland() default threshold: int = 115

# Player icon circle radius detection: 10 ~ 15  
    1. detect_circles: minRadius=10, maxRadius=15