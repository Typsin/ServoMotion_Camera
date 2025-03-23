import socket
import time
import cv2
import numpy as np
import json
import struct
import threading
import os
from datetime import datetime

class QRDetectionClient:
    def __init__(self, host='192.168.3.150', port=5000):
        # Network configuration
        self.host = host
        self.port = port
        self.client_socket = None
        self.is_connected = False
        
        # QR detection setup
        self.qr_detector = cv2.QRCodeDetector()
        self.last_detection = None
        self.last_detection_time = 0
        self.detection_timeout = 1.0  # Seconds to ignore same QR after detection
        
        # Display window
        self.window_name = "QR Detection Client"
        
        # State information from server
        self.current_state = "WAITING_FOR_START"
        self.box1_count = 0
        self.box2_count = 0
        self.current_cycle = 1
        
        # Motion detection parameters
        self.prev_frame = None
        self.motion_threshold = 50  # Threshold for motion detection
        self.stable_frame_count = 0  # Counter for stable frames
        self.min_stable_frames = 3   # Minimum number of stable frames to consider motion stopped
        self.is_stable = False       # Flag to indicate if the camera view is stable
        
        # FPS calculation
        self.fps = 0
        self.frame_count = 0
        self.fps_time = time.time()
        
        # QR code directory 
        self.qr_code_directory = "/home/nash/Capstone_ws/Capstone/okay/QRstuff"

        # Screenshot directory setup
        self.screenshot_dir = "screenshots"
        os.makedirs(self.screenshot_dir, exist_ok=True)
        
        # QR code history to avoid duplicate screenshots
        self.qr_code_history = set()
        
        print(f"Client initialized. Will connect to {host}:{port}")
    
    def connect_to_server(self):
        """Connect to the server"""
        try:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.connect((self.host, self.port))
            self.is_connected = True
            print(f"Connected to server at {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"Connection error: {e}")
            return False
    
    def setup_webcam(self):
        """Initialize the webcam with darker brightness"""
        try:
            self.webcam = cv2.VideoCapture(0)  # Use 0 for default USB webcam
            if not self.webcam.isOpened():
                print("Failed to open webcam")
                return False
                
            # Set resolution (optional)
            self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Reduce brightness (lower values = darker image)
            self.webcam.set(cv2.CAP_PROP_BRIGHTNESS, 0.3)  # Adjust this value (0-1) as needed
            self.webcam.set(cv2.CAP_PROP_CONTRAST, 0.6)    # Slightly increase contrast
            
            print("Webcam initialized successfully with darker brightness")
            return True
        except Exception as e:
            print(f"Webcam setup error: {e}")
            return False

    def send_detection(self, qr_text):
        """Send QR detection to server"""
        try:
            command = {
                "type": "qr_detection",
                "qr_text": qr_text,
                "timestamp": time.time()
            }
            data = json.dumps(command).encode()
            self.client_socket.sendall(struct.pack('!I', len(data)))
            self.client_socket.sendall(data)
            self.qr_code_history.add(qr_text)
            return True
        except Exception as e:
            print(f"Send error: {e}")
            self.is_connected = False
            return False
    
    def send_control_command(self, command):
        """Send control command to server"""
        try:
            # Create command
            cmd_data = {
                "type": "control",
                "command": command,
                "timestamp": time.time()
            }
            
            # Convert to bytes
            data = json.dumps(cmd_data).encode()
            
            # Send size first, then data
            self.client_socket.sendall(struct.pack('!I', len(data)))
            self.client_socket.sendall(data)
            
            print(f"Sent control command: {command}")
            return True
        except Exception as e:
            print(f"Send control error: {e}")
            self.is_connected = False
            return False
    
    def receive_exact_bytes(self, n):
        """Receive exactly n bytes from socket"""
        data = b""
        remaining = n
        while remaining > 0:
            chunk = self.client_socket.recv(min(4096, remaining))
            if not chunk:
                return None
            data += chunk
            remaining -= len(chunk)
        return data
    
    # Remove stability flag checks in detect_qr_code function
    def detect_qr_code(self, frame):
        """Detect QR codes in the frame without stability checks"""
        try:
            # Apply image preprocessing for better detection
            processed_images = self.preprocess_for_qr_detection(frame)
            gray = processed_images['gray']
            thresh = processed_images['thresh']
            
            # Remove stability indicator and related code
            
            # Multi-approach QR detection to handle different orientations
            qr_texts = []
            points_list = []
            
            # Method 1: Standard QR detector on grayscale
            ret_qr, decoded_info, points, _ = self.qr_detector.detectAndDecodeMulti(gray)
            if ret_qr and len(decoded_info) > 0:
                for i, text in enumerate(decoded_info):
                    if text and len(text) > 0:
                        qr_texts.append(text)
                        if points is not None and i < len(points):
                            points_list.append(points[i])
            
            # Method 2: QR detector on thresholded image if Method 1 failed
            if not qr_texts:
                ret_qr, decoded_info, points, _ = self.qr_detector.detectAndDecodeMulti(thresh)
                if ret_qr and len(decoded_info) > 0:
                    for i, text in enumerate(decoded_info):
                        if text and len(text) > 0:
                            qr_texts.append(text)
                            if points is not None and i < len(points):
                                points_list.append(points[i])
            
            # Method 3: Try with rotated images to handle different orientations
            if not qr_texts:
                for angle in [90, 180, 270]:  # Try different rotations
                    # Create rotation matrix
                    height, width = gray.shape[:2]
                    center = (width // 2, height // 2)
                    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                    
                    # Apply rotation to grayscale image
                    rotated = cv2.warpAffine(gray, rotation_matrix, (width, height))
                    
                    # Try detection on rotated image
                    ret_qr, decoded_info, points, _ = self.qr_detector.detectAndDecodeMulti(rotated)
                    
                    if ret_qr and len(decoded_info) > 0:
                        for i, text in enumerate(decoded_info):
                            if text and len(text) > 0:
                                qr_texts.append(text)
                                break
                        
                        if qr_texts:  # If we found a QR code, no need to try more angles
                            break
            
            # Method 4: Try with zbar library if available (more robust to rotation)
            try:
                import pyzbar.pyzbar as pyzbar
                if not qr_texts:
                    decoded_objects = pyzbar.decode(gray)
                    for obj in decoded_objects:
                        qr_texts.append(obj.data.decode('utf-8'))
                        # Get points for visualization
                        points = obj.polygon
                        if points and len(points) > 0:
                            pts = np.array([(p.x, p.y) for p in points])
                            points_list.append(pts)
            except ImportError:
                # pyzbar not available, skip this method
                pass
            
            # Process detected QR codes
            for i, text in enumerate(qr_texts):
                # Normalize text to handle variations
                normalized_text = text.strip().lower()
                
                # Check if this is one of our target QR codes
                is_target = any(target in normalized_text for target in 
                            ['box 1', 'box1', 'box 2', 'box2', 'last and return', 'start'])
                
                # Draw QR code boundary with different color for targets
                if i < len(points_list):
                    pts = points_list[i].astype(int)
                    color = (0, 255, 0) if is_target else (255, 0, 0)  # Green for targets, blue for others
                    frame = cv2.polylines(frame, [pts], True, color, 3)
                    
                    # Add text above QR code with better visibility
                    text_bg_pt1 = (pts[0][0], pts[0][1] - 30)
                    text_bg_pt2 = (pts[0][0] + max(100, len(text)*10), pts[0][1])
                    frame = cv2.rectangle(frame, text_bg_pt1, text_bg_pt2, (0, 0, 0), -1)
                    cv2.putText(frame, text, (pts[0][0], pts[0][1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Check if this is a new detection or within timeout
                current_time = time.time()
                should_process = (text != self.last_detection or 
                                current_time - self.last_detection_time > self.detection_timeout)
                
                # Don't process start QR codes if we're already in searching mode
                if "start" in normalized_text and self.current_state == "SEARCHING_BOXES":
                    print(f"Ignoring start QR in SEARCHING_BOXES state: {text}")
                    should_process = False
                
                if should_process:
                    print(f"QR code detected: {text}")
                    self.last_detection = text
                    self.last_detection_time = current_time
                    
                    # Send detection to server
                    self.send_detection(text)
            
            return frame
        except Exception as e:
            print(f"QR detection error: {e}")
            return frame
    
    def preprocess_for_qr_detection(self, frame):
        """Enhance image for better QR code detection"""
        try:
            # Create a copy of the frame
            processed = frame.copy()
            
            # Convert to grayscale
            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            
            # Apply bilateral filter to reduce noise while preserving edges
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Apply contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(filtered)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Apply morphological operations
            kernel = np.ones((3, 3), np.uint8)
            morphed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            # Return processed images for QR detection
            return {
                'original': frame,
                'gray': gray,
                'filtered': filtered,
                'enhanced': enhanced,
                'thresh': thresh, 
                'morphed': morphed
            }
        except Exception as e:
            print(f"Image preprocessing error: {e}")
            return {'original': frame, 'gray': cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)}
    
    def process_frame(self, frame):
        """Add visualization elements to the frame and perform QR detection"""
        if frame is None or frame.size == 0:
            print("Skipping process_frame for invalid frame")
            return None        
        
        # Detect QR codes first
        frame = self.detect_qr_code(frame)
        
        # Check server state and add visual indicator
        if self.current_state == "SEARCHING_BOXES":
            status_color = (0, 255, 0)  # Green for active scanning
        elif self.current_state == "RETURNING":
            status_color = (0, 255, 255)  # Yellow for returning
        elif self.current_state == "CYCLE_COMPLETE":
            status_color = (0, 0, 255)  # Red for cycle complete
        else:
            status_color = (255, 255, 255)  # White for waiting
        
        # Draw state information with colored background
        cv2.rectangle(frame, (5, 5), (250, 35), (0, 0, 0), -1)  # Black background
        cv2.putText(frame, f"State: {self.current_state}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Draw counts
        cv2.rectangle(frame, (5, 40), (250, 95), (0, 0, 0), -1)  # Black background
        cv2.putText(frame, f"Box1 Count: {self.box1_count}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Box2 Count: {self.box2_count}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw FPS and cycle info
        cv2.rectangle(frame, (5, 100), (250, 155), (0, 0, 0), -1)  # Black background
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Cycle: {self.current_cycle}", (10, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Add control instructions
        cv2.rectangle(frame, (frame.shape[1] - 210, 5), (frame.shape[1] - 5, 35), (0, 0, 0), -1)
        cv2.putText(frame, "Press 'q' to quit", (frame.shape[1] - 200, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def receive_data(self):
        """Receive and process data from server with enhanced error handling"""
        try:
            # Receive data size (4 bytes)
            size_data = self.receive_exact_bytes(4)
            if not size_data:
                print("Connection closed by server")
                return None, None
            
            size = struct.unpack('!I', size_data)[0]
            
            # Try to peek at the first few bytes to check if it's JSON
            peek_data = self.client_socket.recv(min(64, size), socket.MSG_PEEK)
            if peek_data.startswith(b'{'):
                # This might be JSON data
                data = self.receive_exact_bytes(size)
                try:
                    json_data = json.loads(data.decode())
                    
                    # Check data type
                    if "type" in json_data:
                        if json_data["type"] == "screenshot":
                            # This is a screenshot, receive the image data
                            img_size = json_data["size"]
                            qr_type = json_data["qr_type"]
                            timestamp = json_data["timestamp"]
                            
                            # Get the QR text if available
                            qr_text = json_data.get("qr_text", "unknown")
                            
                            # Only process screenshot if motion has stopped
                            if not self.is_stable:
                                print("Skipping screenshot processing due to motion")
                                # Skip receiving image data if we're not going to use it
                                self.receive_exact_bytes(img_size)
                                return "skipped", None
                            
                            # Check if we've already processed this QR code
                            if qr_text in self.qr_code_history:
                                print(f"Already processed QR code: {qr_text}")
                                # Skip receiving image data if we've already processed this QR
                                self.receive_exact_bytes(img_size)
                                return "skipped", None
                            
                            image_data = self.receive_exact_bytes(img_size)
                            
                            # Convert to numpy array and decode
                            try:
                                img = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                                
                                # Save the image with QR text in filename (sanitize filename)
                                safe_qr_text = "".join([c if c.isalnum() else "_" for c in qr_text])
                                filename = f"{self.screenshot_dir}/{qr_type}_{safe_qr_text}_{timestamp}.jpg"
                                cv2.imwrite(filename, img)
                                print(f"Screenshot saved: {filename}")
                                
                                return "screenshot", img
                            except Exception as e:
                                print(f"Screenshot decode error: {e}")
                                return None, None
                        
                        elif json_data["type"] == "counts":
                            # Update the counts in real-time
                            self.box1_count = json_data["box_count"]
                            self.box2_count = json_data["box2_count"]
                            self.current_state = json_data["current_state"]
                            self.current_cycle = json_data["current_cycle"]
                            print(f"Updated counts received: Box1={self.box1_count}, Box2={self.box2_count}")
                            return "counts", json_data
                except json.JSONDecodeError:
                    # Not valid JSON, treat as regular frame
                    pass
            
            # Default behavior: treat as a regular video frame
            frame_data = self.receive_exact_bytes(size)
            

            try:
                # Convert to numpy array and decode    
                frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None or frame.size == 0:
                    print("Received invalid frame data, skipping frame")
                    return None, None

                # Update FPS counter
                self.frame_count += 1
                if time.time() - self.fps_time >= 1.0:
                    self.fps = self.frame_count / (time.time() - self.fps_time)
                    self.frame_count = 0
                    self.fps_time = time.time()
                
                return "frame", frame
            except Exception as e:
                print(f"Frame decode error: {e}")
                return None, None

        except Exception as e:
            print(f"Data receive error: {e}")
            return None, None

    def run(self):
        """Main execution function"""
        # Connect to server
        if not self.connect_to_server():
            print("Failed to connect to server. Exiting.")
            return
        
        # Create display window
        cv2.namedWindow(self.window_name)
        
        try:
            # Main processing loop
            while self.is_connected:
                # Receive data from server
                data_type, data = self.receive_data()
                
                if data_type == "frame":
                    # Process frame (detect QR codes, add visualization)
                    frame = self.process_frame(data)
                    
                    # Display frame
                    cv2.imshow(self.window_name, frame)
                    
                    # Check for key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("User requested exit")
                        break
                    
                elif data_type == "screenshot":
                    # Screenshot already saved, just display
                    if data is not None:
                        cv2.imshow("Screenshot", data)
                        cv2.waitKey(1000)  # Display for 1 second
                        cv2.destroyWindow("Screenshot")
                    
                elif data_type == "counts":
                    # Counts already updated in receive_data
                    pass
                    
                elif data_type == "skipped":
                    # Data was deliberately skipped, do nothing
                    pass
                    
                elif data_type is None:
                    # Connection lost or error
                    print("Connection lost. Exiting.")
                    break
                    
        except KeyboardInterrupt:
            print("Client terminated by user")
        except Exception as e:
            print(f"Client error: {e}")
        finally:
            # Cleanup
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up resources")
        
        # Close socket
        if self.client_socket:
            self.client_socket.close()
        
        # Close window
        cv2.destroyAllWindows()
        
        print("Cleanup complete")

    def tune_motion_sensitivity(self, threshold=None, min_stable_frames=None):
        """Tune motion detection sensitivity parameters"""
        if threshold is not None:
            self.motion_threshold = threshold
            print(f"Motion threshold set to {threshold}")
        
        if min_stable_frames is not None:
            self.min_stable_frames = min_stable_frames
            print(f"Minimum stable frames set to {min_stable_frames}")

if __name__ == "__main__":
    # You can set the IP address of the Raspberry Pi here
    client = QRDetectionClient(host='192.168.3.150', port=5000)
        
    # Optional: Adjust sensitivity parameters
    # Lower threshold = more sensitive to motion
    # Higher min_stable_frames = require more stable frames before considering "stable"
    client.tune_motion_sensitivity(threshold=50, min_stable_frames=3)
        
    client.run()