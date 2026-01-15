import bettercam
import onnxruntime as ort
import numpy as np
import cv2
import dearpygui.dearpygui as dpg
import threading
import time
import math
from pynput import keyboard

# Windows API for True Transparency
import win32gui
import win32con
import win32api

# ================= CONFIGURATION =================
MODEL_PATH = "best.onnx"    # Reverting to 640 (Model architecture requires this)
INPUT_SIZE = 640              # Reverting to 640
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

# Default Zone Settings
DEFAULT_ZONE_W = 100
DEFAULT_ZONE_H = 100

TRIGGER_KEY = 'q'             # Key to press
HOLD_KEY = 'c'                # Key to HOLD to activate bot ('c' is common, user can change)
TRIGGER_COOLDOWN = 0.1

# ================= KEYBOARD CONTROLLER =================
kb_controller = keyboard.Controller()

# ================= UTILS =================
def xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def nms(boxes, scores, iou_threshold):
    if len(boxes) == 0: return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

# ================= BOT STATE & THREADS =================
class SquareBot:
    def __init__(self):
        self.running = True
        
        # Settings
        self.zone_w = DEFAULT_ZONE_W
        self.zone_h = DEFAULT_ZONE_H
        
        # Dynamic Keys
        self.trigger_key = TRIGGER_KEY
        self.hold_key = HOLD_KEY
        
        # Visual Toggles
        self.show_hitbox = True
        self.show_line = True
        self.show_center = True
        self.show_box = True
        self.show_menu = True # Toggle for Cheat Menu
        
        # State
        self.active_hold = False
        self.last_trigger_time = 0
        
        # Data
        self.latest_frame = None
        self.detections = []      # list of (cx, cy, x1, y1, x2, y2)
        self.prediction_line = None # ((start_x, start_y), (end_x, end_y))
        self.fps = 0
        
        # Movement Tracking
        self.history_pos = []     # List of (cx, cy, time)
        
        # 1. Camera
        try:
            self.camera = bettercam.create(output_idx=0, output_color="BGRA")
            test = self.camera.grab()
            if test is None: raise Exception("Camera grab failed")
            self.screen_h, self.screen_w = test.shape[:2]
            self.center_x = self.screen_w // 2
            self.center_y = self.screen_h // 2
            print(f"[INFO] Camera OK: {self.screen_w}x{self.screen_h}")
        except Exception as e:
            print(f"[CRITICAL] Camera Init Fail: {e}")
            exit()

        # 2. Model
        print(f"[INFO] Loading Model: {MODEL_PATH} (Input: {INPUT_SIZE}x{INPUT_SIZE})")
        
        # Priority: CUDA (If installed) -> DirectML (Friendly GPU) -> CPU
        providers = ['CUDAExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider']
        
        try:
            # Check providers
            avail = ort.get_available_providers()
            print(f"[INFO] Available ONNX Providers: {avail}")
            
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=providers)
            
            used_provider = self.session.get_providers()[0]
            print(f"[INFO] Active Provider: {used_provider}")
            
            if used_provider == 'CPUExecutionProvider':
                 print("[WARNING] Running on CPU! Install 'onnxruntime-directml' for easy GPU support.")
            elif used_provider == 'DmlExecutionProvider':
                 print("[SUCCESS] Using DirectML (GPU) - Driver Friendly Mode!")
            
        except Exception as e:
            print(f"[CRITICAL] Model Load Fail: {e}")
            print("TRY: python fix_model_opset.py")
            exit()
            
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def create_menu_theme(self):
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvAll):
                # Cheat Menu Style: Dark Grey + Vibrant Purple/Cyan Accents
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (20, 20, 20, 240))
                dpg.add_theme_color(dpg.mvThemeCol_Border, (140, 0, 255, 150))
                dpg.add_theme_color(dpg.mvThemeCol_Button, (70, 70, 70, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (140, 0, 255, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (180, 50, 255, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Text, (220, 220, 220, 255))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (40, 40, 40, 255))
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, (140, 0, 255, 255))
                dpg.add_theme_color(dpg.mvThemeCol_CheckMark, (0, 255, 255, 255))
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 6)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
                dpg.add_theme_style(dpg.mvStyleVar_WindowBorderSize, 1)
        return theme

    def create_transparent_theme(self):
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvAll):
                # Force everything to be transparent by default unless specified
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (0, 0, 0, 0))
                dpg.add_theme_color(dpg.mvThemeCol_Border, (0, 0, 0, 0))
        return theme

    def make_window_transparent(self, window_title):
        """Use Windows API to make the DPG viewport truly transparent."""
        # Find the window by title
        hwnd = win32gui.FindWindow(None, window_title)
        if hwnd == 0:
            print(f"[WARN] Could not find window: {window_title}")
            return False
        
        # Get current extended style
        ex_style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
        
        # Add ONLY Layered style (NOT WS_EX_TRANSPARENT, or menu becomes unclickable!)
        new_ex_style = ex_style | win32con.WS_EX_LAYERED
        win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, new_ex_style)
        
        # Set the transparent color key (BLACK = fully transparent)
        # LWA_COLORKEY = 0x00000001
        # LWA_ALPHA = 0x00000002
        # Using Black (0,0,0) as the color key - anything black will be invisible
        win32gui.SetLayeredWindowAttributes(hwnd, win32api.RGB(0, 0, 0), 0, win32con.LWA_COLORKEY)
        
        print(f"[SUCCESS] Window '{window_title}' is now truly transparent!")
        return True

    def press_trigger(self):
        kb_controller.press(self.trigger_key)
        time.sleep(0.04) 
        kb_controller.release(self.trigger_key)

    def start(self):
        # Key Listener
        self.listener = keyboard.Listener(on_press=self.on_key_press, on_release=self.on_key_release)
        self.listener.start()
        
        # Threads
        t1 = threading.Thread(target=self.capture_loop, daemon=True)
        t2 = threading.Thread(target=self.inference_loop, daemon=True)
        t1.start()
        t2.start()
        
        # GUI (Main Thread)
        self.run_gui()

    def on_key_press(self, key):
        try:
            if key == keyboard.Key.insert:
                self.show_menu = not self.show_menu
                
            if hasattr(key, 'char') and key.char == self.hold_key:
                self.active_hold = True
        except AttributeError:
            pass

    def on_key_release(self, key):
        try:
            if hasattr(key, 'char') and key.char == self.hold_key:
                self.active_hold = False
        except AttributeError:
            pass

    def capture_loop(self):
        while self.running:
            frame = self.camera.grab()
            if frame is not None:
                self.latest_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            else:
                time.sleep(0.001)

    def inference_loop(self):
        frame_counter = 0
        start_t = time.time()
        
        while self.running:
            if self.latest_frame is None:
                time.sleep(0.005)
                continue

            # --- Logic: If NOT holding key, sleep to save resources ---
            if not self.active_hold:
                # Clear state when idle
                self.detections = []
                # self.prediction_line = None # Keep outline visible if wanted? Nah.
                self.history_pos = []
                time.sleep(0.05) 
                # continue # Remove continue to keep rendering GUI even when idle?
                # Actually, inference loop doesn't render GUI. GUI is separate.
                # But if we continue here, we don't update self.detections.
                
                # To clarify: The user wants to see overlay.
                # If we skip inference, we see nothing.
                # Let's run a light loop or just skip inference but keep loop alive?
                # For "Hold to Active", we usually want NO scanning when idle.
                # So clearing detections is correct.
                # GUI thread will just draw nothing or just the crosshair.
                continue

            # 1. Preprocess
            img = self.latest_frame.copy()
            img_resized = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_in = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
            img_in = np.expand_dims(img_in, axis=0)

            # 2. Inference
            try:
                outputs = self.session.run([self.output_name], {self.input_name: img_in})
            except Exception as e:
                print(f"[ERROR] Inference failed: {e}")
                print("[INFO] Attempting to fallback to CPU...")
                try:
                    self.session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
                    self.input_name = self.session.get_inputs()[0].name
                    self.output_name = self.session.get_outputs()[0].name
                    print("[SUCCESS] Switched to CPU mode! Continuing...")
                    continue
                except Exception as cpu_e:
                    print(f"[CRITICAL] CPU Fallback failed: {cpu_e}")
                    self.running = False
                    break

            # 3. Post-process
            prediction = np.squeeze(outputs[0]).T
            scores = np.max(prediction[:, 4:], axis=1) if prediction.shape[1] > 4 else prediction[:, 4]
            mask = scores > CONF_THRESHOLD
            prediction = prediction[mask]
            scores = scores[mask]

            detected = []
            if len(prediction) > 0:
                boxes = xywh2xyxy(prediction[:, :4])
                indices = nms(boxes, scores, IOU_THRESHOLD)
                
                scale_x = self.screen_w / INPUT_SIZE
                scale_y = self.screen_h / INPUT_SIZE

                for i in indices:
                    box = boxes[i]
                    x1 = int(box[0] * scale_x)
                    y1 = int(box[1] * scale_y)
                    x2 = int(box[2] * scale_x)
                    y2 = int(box[3] * scale_y)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    detected.append((cx, cy, x1, y1, x2, y2))
            
            self.detections = detected

            # 4. Trigger & Prediction Logic
            if detected:
                # Get closest shadow to center
                target = min(detected, key=lambda d: math.hypot(d[0]-self.center_x, d[1]-self.center_y))
                tx, ty, _, _, _, _ = target
                
                # Update history for prediction
                now = time.time()
                self.history_pos.append((tx, ty, now))
                if len(self.history_pos) > 5: self.history_pos.pop(0)
                
                # Calculate Vector
                if len(self.history_pos) >= 2:
                    dx_total = self.history_pos[-1][0] - self.history_pos[0][0]
                    dy_total = self.history_pos[-1][1] - self.history_pos[0][1]
                    dt = self.history_pos[-1][2] - self.history_pos[0][2]
                    
                    if dt > 0:
                        vx = dx_total / dt # pixels per sec
                        vy = dy_total / dt
                        # Predict position in 0.5s
                        pred_x = int(tx + vx * 0.5)
                        pred_y = int(ty + vy * 0.5)
                        self.prediction_line = ((tx, ty), (pred_x, pred_y))
                else:
                    self.prediction_line = None

                # Square Hitbox Check
                dx = abs(tx - self.center_x)
                dy = abs(ty - self.center_y)
                
                half_w = self.zone_w / 2
                half_h = self.zone_h / 2
                
                if dx < half_w and dy < half_h:
                    if now - self.last_trigger_time > TRIGGER_COOLDOWN:
                        # ACTION!
                        print(f"[TRIGGER] Target in Square Zone! ({dx:.0f}x{dy:.0f})")
                        threading.Thread(target=self.press_trigger).start()
                        self.last_trigger_time = now

            # FPS
            frame_counter += 1
            if time.time() - start_t > 1:
                self.fps = frame_counter
                frame_counter = 0
                start_t = time.time()

    def run_gui(self):
        try:
            dpg.create_context()
            # Viewport: The OS Window (Transparent, Always on Top, Borderless)
            dpg.create_viewport(title="Legend Ware v2", width=self.screen_w, height=self.screen_h, 
                                always_on_top=True, decorated=False, clear_color=(0,0,0,0))
            dpg.setup_dearpygui()
            
            # Prepare Themes
            menu_theme = self.create_menu_theme()
            trans_theme = self.create_transparent_theme()
            
            # 1. BIND GLOBAL TRANSPARENT THEME
            dpg.bind_theme(trans_theme)
            
            # 2. ENFORCE VIEWPORT TRANSPARENCY
            # (Important for some Windows versions)
            dpg.set_viewport_clear_color((0, 0, 0, 0))

            # 3. Overlay Layer (Passthrough)
            with dpg.window(label="Overlay", tag="overlay", width=self.screen_w, height=self.screen_h, pos=(0,0),
                            no_title_bar=True, no_resize=True, no_move=True, no_background=True, 
                            no_bring_to_front_on_focus=True):
                # No specific theme bound, uses Global Transparent
                dpg.add_draw_layer(tag="draw_layer")

            # 4. Cheat Menu (Toggleable) - BIND DARK THEME HERE
            with dpg.window(label="LEGEND WARE [INS]", tag="menu_win", width=400, height=500, pos=(50, 50), no_collapse=True):
                dpg.bind_item_theme("menu_win", menu_theme) # Apply Dark Theme ONLY to Menu
                
                with dpg.group(horizontal=True):
                    dpg.add_text("STATUS: ")
                    self.status_text = dpg.add_text("IDLE", color=(255, 0, 0))
                
                with dpg.tab_bar():
                    # TAB 1: AIMBOT (Zone & Keys)
                    with dpg.tab(label="Aimbot"):
                        dpg.add_spacer(height=5)
                        dpg.add_text("Trigger Zone", color=(0, 255, 255))
                        dpg.add_slider_int(label="Width", default_value=DEFAULT_ZONE_W, min_value=10, max_value=800, 
                                           callback=lambda s, a: setattr(self, 'zone_w', a))
                        dpg.add_slider_int(label="Height", default_value=DEFAULT_ZONE_H, min_value=10, max_value=800, 
                                           callback=lambda s, a: setattr(self, 'zone_h', a))
                        
                        dpg.add_spacer(height=10)
                        dpg.add_text("Keybinds", color=(0, 255, 255))
                        with dpg.group(horizontal=True):
                            dpg.add_text("Hold Key:   ")
                            dpg.add_input_text(default_value=self.hold_key, width=50, 
                                               callback=lambda s, a: setattr(self, 'hold_key', a.lower() if a else 'c'))
                        
                        with dpg.group(horizontal=True):
                            dpg.add_text("Trigger Key:")
                            dpg.add_input_text(default_value=self.trigger_key, width=50, 
                                               callback=lambda s, a: setattr(self, 'trigger_key', a.lower() if a else 'q'))

                    # TAB 2: VISUALS (ESP)
                    with dpg.tab(label="Visuals"):
                        dpg.add_spacer(height=5)
                        dpg.add_text("ESP Options", color=(0, 255, 255))
                        dpg.add_checkbox(label="Show Hitbox (Red)", default_value=True, 
                                        callback=lambda s, a: setattr(self, 'show_hitbox', a))
                        dpg.add_checkbox(label="Show Prediction (Line)", default_value=True, 
                                        callback=lambda s, a: setattr(self, 'show_line', a))
                        dpg.add_checkbox(label="Show Crosshair (Green)", default_value=True, 
                                        callback=lambda s, a: setattr(self, 'show_center', a))
                        dpg.add_checkbox(label="Show ESP Box (Cyan)", default_value=True, 
                                        callback=lambda s, a: setattr(self, 'show_box', a))

                    # TAB 3: SETTINGS
                    with dpg.tab(label="Settings"):
                        dpg.add_spacer(height=5)
                        dpg.add_text("Performance", color=(0, 255, 255))
                        dpg.add_text("FPS: 0", tag="fps_lbl")
                        dpg.add_text(f"Device: {self.session.get_providers()[0]}")
                        dpg.add_button(label="Force Exit Bot", callback=lambda: dpg.stop_dearpygui(), width=-1)

            # Show the Viewport
            dpg.show_viewport()
            # Maximize to ensure overlay covers everything
            dpg.maximize_viewport()
            
            # WINDOWS API: Make the window TRULY transparent
            time.sleep(0.1) # Wait a moment for window to be ready
            self.make_window_transparent("Legend Ware v2")

            while dpg.is_dearpygui_running():
                # Toggle Menu Visibility
                if self.show_menu:
                    dpg.configure_item("menu_win", show=True)
                else:
                    dpg.configure_item("menu_win", show=False)

                # Status update
                if self.active_hold:
                     dpg.set_value(self.status_text, f"ACTIVE [{self.hold_key.upper()}]")
                     dpg.configure_item(self.status_text, color=(0, 255, 0))
                else:
                     dpg.set_value(self.status_text, f"IDLE [{self.hold_key.upper()}]")
                     dpg.configure_item(self.status_text, color=(255, 0, 0))
                     
                dpg.set_value("fps_lbl", f"FPS: {self.fps}")

                # Draw Overlay
                dpg.delete_item("draw_layer", children_only=True)
                
                # Draw Kill Zone (Red Square)
                if self.show_hitbox:
                    half_w = int(self.zone_w / 2)
                    half_h = int(self.zone_h / 2)
                    dpg.draw_rectangle((self.center_x - half_w, self.center_y - half_h), 
                                     (self.center_x + half_w, self.center_y + half_h), 
                                     color=(255, 0, 0, 200), thickness=2, parent="draw_layer")
                
                # Draw Center (Crosshair)
                if self.show_center:
                    dpg.draw_line((self.center_x - 10, self.center_y), (self.center_x + 10, self.center_y), color=(0, 255, 0), thickness=2, parent="draw_layer")
                    dpg.draw_line((self.center_x, self.center_y - 10), (self.center_x, self.center_y + 10), color=(0, 255, 0), thickness=2, parent="draw_layer")

                # Draw Detections
                for (cx, cy, x1, y1, x2, y2) in self.detections:
                    if self.show_box:
                        dpg.draw_rectangle((x1, y1), (x2, y2), color=(0, 255, 255), thickness=1, parent="draw_layer")
                    if self.show_center:
                        dpg.draw_circle((cx, cy), 3, color=(255, 255, 0), fill=(255, 255, 0), parent="draw_layer")
                
                # Draw Prediction
                if self.show_line and self.prediction_line:
                    dpg.draw_line(self.prediction_line[0], self.prediction_line[1], color=(255, 255, 0), thickness=2, parent="draw_layer")
                    dpg.draw_circle(self.prediction_line[1], 4, color=(255, 0, 255), parent="draw_layer")

                dpg.render_dearpygui_frame()
                
        except Exception as e:
            print(f"[GUI ERROR] {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.running = False
            self.listener.stop()
            if dpg.is_dearpygui_running():
                dpg.destroy_context()

if __name__ == "__main__":
    bot = SquareBot()
    bot.start()
