import socket
import time
import cv2
import numpy as np
import json
import struct
import threading
import RPi.GPIO as GPIO

# GPIO Assignments
MG996R_PIN = 18  # GPIO 18 (Pin 12) for MG996R servo
L298N_IN1 = 17   # GPIO 17 (Pin 11) for L298N IN1
L298N_IN2 = 27   # GPIO 27 (Pin 13) for L298N IN2
L298N_ENA = 22   # GPIO 22 (Pin 15) for L298N ENA (PWM)
ENC_A = 5        # GPIO 5 (Pin 29) for Encoder A
ENC_B = 6        # GPIO 6 (Pin 31) for Encoder B

class QRServerPi:
    def __init__(self, host='0.0.0.0', port=5000):
        # Network configuration
        self.host = host
        self.port = port
        self.server_socket = None
        self.client_socket = None
        self.is_connected = False
        
        # Webcam setup
        self.webcam = None
        self.is_streaming = False
        self.stream_thread = None
        self.max_fps = 20  # Limit FPS to reduce network load
        self.last_frame_time = 0    
        
        # QR detection state
        self.current_state = "WAITING_FOR_START"
        self.box1_count = 0
        self.box2_count = 0
        self.current_cycle = 1
        self.last_qr_detected = None
        self.last_qr_time = 0
        self.qr_timeout = 2.0  # Seconds to ignore same QR after detection
        self.qr_ignore_until = 0  # Timestamp until which QR codes should be ignore
        
        #Motor control
        self.motor_thread = None
        self.motor_command = None
        self.should_stop = False
        self.motor_moving = False  # Track if motors are currently moving
        self.motor_last_moved = 0  # Time when motor last stopped moving
        
        # Stepped movement control
        self.step_duration = 0.3    # Motor movement time per step
        self.pause_duration = 0.7   # Pause time after each step
        self.max_steps = 20         # Maximum steps before timeout

        # QR code directory 
        self.qr_code_directory = "/home/rpi/capstone/Pray/QRstuff"

    def process_qr_detection(self, qr_text):
            """Process QR code detection with timing controls"""
            normalized_text = qr_text.strip().lower()
            current_time = time.time()
            
            # Check motor activity and cooldown period
            motor_stopped_long_enough = not self.motor_moving and (current_time - self.motor_last_moved >= self.scan_pause_time)
            
            # Ignore QR if motor is active or in cooldown (unless waiting for start)
            if not motor_stopped_long_enough and self.current_state != "WAITING_FOR_START":
                print(f"Motor moving/recently moved - ignoring QR: {normalized_text}")
                return
            
            # Ignore duplicate or cooldown QRs
            if normalized_text == self.last_qr_detected and current_time - self.last_qr_time < self.qr_timeout:
                print(f"Duplicate QR ignored: {normalized_text}")
                return
            
            # Check if in box QR cooldown period
            if current_time < self.ignore_boxes_until:
                print(f"Ignoring box QR during cooldown: {normalized_text}")
                return
            
            self.last_qr_detected = normalized_text
            self.last_qr_time = current_time
            
            print(f"QR detected: '{normalized_text}' in state {self.current_state}")
            
            # State machine logic
            if self.current_state == "WAITING_FOR_START":
                if "start" in normalized_text and str(self.current_cycle) in normalized_text:
                    print(f"Start {self.current_cycle} detected: Beginning cycle")
                    self.current_state = "SEARCHING_BOXES"
                    self.motor_command = "mg996r_clockwise"  # Continuous movement
                    self.take_screenshot("start", normalized_text)
                    self.ignore_boxes_until = time.time() + 1  # Ignore boxes for 1s
                    
            elif self.current_state == "SEARCHING_BOXES":
                if "box 1" in normalized_text:
                    print("Box 1 detected: Stopping for 5s")
                    self.box1_count += 1
                    self.motor_command = "mg996r_stop"
                    self.take_screenshot("box1", normalized_text)
                    time.sleep(5.0)  # Wait 5 seconds
                    self.motor_command = "mg996r_clockwise"
                    self.ignore_boxes_until = time.time() + 1  # Post-movement cooldown
                    
                elif "box 2" in normalized_text:
                    print("Box 2 detected: Stopping for 5s")
                    self.box2_count += 1
                    self.motor_command = "mg996r_stop"
                    self.take_screenshot("box2", normalized_text)
                    time.sleep(5.0)  # Wait 5 seconds
                    self.motor_command = "mg996r_clockwise"
                    self.ignore_boxes_until = time.time() + 1
                    
                elif "last and return" in normalized_text:
                    print("Last & Return detected: Reversing")
                    self.current_state = "RETURNING"
                    self.motor_command = "mg996r_anticlockwise"
            
            elif self.current_state == "RETURNING":
                if "start" in normalized_text and str(self.current_cycle) in normalized_text:
                    print(f"Returned to Start {self.current_cycle}")
                    self.motor_command = "mg996r_stop"
                    self.current_state = "CYCLE_COMPLETE"
                    threading.Timer(5.0, self.check_next_cycle).start()
            
            # Update client with new counts
            self.send_data("counts", None)

    def stepped_movement_search(self):
        """Move servo in steps while searching for boxes with explicit motor commands"""
        print("Starting stepped movement search with intermittent clockwise motion")
        self.current_state = "SEARCHING_BOXES"
        steps = 0
        qr_ignore_time = 0
        
        while self.current_state == "SEARCHING_BOXES" and steps < self.max_steps:
            # Move clockwise
            self.motor_command = "mg996r_clockwise"
            self.mg996r_clock()  # Directly call motor function
            time.sleep(self.step_duration)
            
            # Stop the motor
            self.motor_command = "mg996r_stop"
            self.mg996r_stop()  # Directly call stop function
            
            # Set QR ignore time when motor stops
            qr_ignore_time = time.time() + 1.0  # Ignore QR codes for 1 second
            
            # Pause for detection
            time.sleep(self.pause_duration)
            
            steps += 1
            print(f"Search step {steps}/{self.max_steps} completed")
        
        if steps >= self.max_steps:
            print("Search timeout, no QR detected")
            self.motor_command = "mg996r_stop"
            self.mg996r_stop()

    def process_qr_detection(self, qr_text):
        """Process QR code detection with improved handling for box 1 and box 2"""
        # Normalize QR text (remove extra spaces, lowercase)
        normalized_text = qr_text.strip().lower()
        
        # Check if we're currently ignoring QR codes (after movement)
        current_time = time.time()
        if hasattr(self, 'qr_ignore_until') and current_time < self.qr_ignore_until:
            print(f"Ignoring QR in cool-down period: {normalized_text}")
            return
        
        # Ignore rapid consecutive detections of the same QR code
        if normalized_text == self.last_qr_detected and current_time - self.last_qr_time < self.qr_timeout:
            print(f"Duplicate QR detection ignored: {normalized_text}")
            return
        
        # Don't process start QR codes if we're already in searching mode
        if "start" in normalized_text and self.current_state == "SEARCHING_BOXES":
            print(f"Ignoring start QR in SEARCHING_BOXES state: {normalized_text}")
            return
        
        # Look for specific QR codes from /home/rpi/capstone/Pray/QRstuff
        # Use the specific QR codes from the directory
        specific_qr_path = "/home/rpi/capstone/Pray/QRstuff"
        
        self.last_qr_detected = normalized_text
        self.last_qr_time = current_time
        
        print(f"QR detected: '{normalized_text}' in state {self.current_state}")
        
        # State machine for QR detection sequence
        if self.current_state == "WAITING_FOR_START":
            if "start" in normalized_text and str(self.current_cycle) in normalized_text:
                print(f"Start {self.current_cycle} QR detected, beginning cycle")
                self.current_state = "SEARCHING_BOXES"
                
                # Take screenshot for start QR
                self.take_screenshot("start", normalized_text)
                
                # Force immediate motor movement - explicitly call the function
                self.mg996r_stop()  # First ensure motor is stopped
                time.sleep(0.2)
                self.mg996r_clock()  # Start clockwise movement explicitly
                
                # Start the stepped movement search in a separate thread
                threading.Thread(target=self.stepped_movement_search).start()
                
        elif self.current_state == "SEARCHING_BOXES":
            if "box 1" in normalized_text or "box1" in normalized_text:
                print("Box 1 detected, stopping for 3 seconds")
                self.motor_command = "mg996r_stop"
                self.mg996r_stop()
                
                # Capture and send screenshot
                ret, frame = self.webcam.read()
                if ret:
                    self.send_data("screenshot", frame, {
                        "qr_type": "box1",
                        "qr_text": normalized_text
                    })
                
                # Update counts
                self.box1_count += 1
                
                # Stop for 3 seconds
                time.sleep(3.0)
                
                # Resume movement and set ignore time
                self.qr_ignore_until = time.time() + 1.0  # Ignore QR for 1 second after resuming
                threading.Thread(target=self.stepped_movement_search).start()
                
            elif "box 2" in normalized_text or "box2" in normalized_text:
                print("Box 2 detected, stopping for 3 seconds")
                self.motor_command = "mg996r_stop"
                self.mg996r_stop()
                
                # Capture and send screenshot
                ret, frame = self.webcam.read()
                if ret:
                    self.send_data("screenshot", frame, {
                        "qr_type": "box2",
                        "qr_text": normalized_text
                    })
                
                # Update counts
                self.box2_count += 1
                
                # Stop for 3 seconds
                time.sleep(3.0)
                
                # Resume movement and set ignore time
                self.qr_ignore_until = time.time() + 1.0  # Ignore QR for 1 second after resuming
                threading.Thread(target=self.stepped_movement_search).start()
                
            elif "last and return" in normalized_text:
                print("Last & Return QR detected, reversing direction")
                self.current_state = "RETURNING"
                threading.Thread(target=self.stepped_movement_return).start()

        elif self.current_state == "RETURNING":
            if qr_text == f"start {self.current_cycle}":
                print(f"Returned to Start {self.current_cycle}")
                self.motor_command = "mg996r_stop"
                self.current_state = "CYCLE_COMPLETE"
                self.current_cycle += 1
                threading.Timer(5.0, self.check_next_cycle).start()
        
        # Send updated counts to client
        self.send_data("counts", None)

        
    def setup_gpio(self):
        """Initialize GPIO pins for motors and encoder"""
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(MG996R_PIN, GPIO.OUT)
        GPIO.setup(L298N_IN1, GPIO.OUT)
        GPIO.setup(L298N_IN2, GPIO.OUT)
        GPIO.setup(L298N_ENA, GPIO.OUT)
        GPIO.setup(ENC_A, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(ENC_B, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        
        # Add encoder interrupt
        GPIO.add_event_detect(ENC_A, GPIO.BOTH, callback=self.encoder_callback)
        
        # PWM Setup
        self.mg996r_pwm = GPIO.PWM(MG996R_PIN, 50)  # 50 Hz for MG996R
        self.l298n_pwm = GPIO.PWM(L298N_ENA, 1000)  # 1000 Hz for L298N
        self.mg996r_pwm.start(0)
        self.l298n_pwm.start(0)
        
        print("GPIO setup completed")
    
    def encoder_callback(self, channel):
        """Callback function for encoder interrupts"""
        if GPIO.input(ENC_A) == GPIO.input(ENC_B):
            self.encoder_count += 1
        else:
            self.encoder_count -= 1
    
    def mg996r_clock(self):
        """Rotate the MG996R servo clockwise at reduced speed"""
        self.mg996r_pwm.ChangeDutyCycle(2.5)  # 2.5% duty cycle for clockwise rotation
        self.motor_moving = True
    
    def mg996r_anti(self):
        """Rotate the MG996R servo anti-clockwise at reduced speed"""
        self.mg996r_pwm.ChangeDutyCycle(7.5)  # 7.5% duty cycle for anti-clockwise rotation
        self.motor_moving = True
    
    def mg996r_stop(self):
        """Stop the MG996R servo"""
        self.mg996r_pwm.ChangeDutyCycle(0)
        self.motor_moving = False
        self.motor_last_moved = time.time()
    
    def mgm_forward(self):
        """Move the metal gearmotor forward at reduced speed"""
        GPIO.output(L298N_IN1, GPIO.LOW)
        GPIO.output(L298N_IN2, GPIO.HIGH)
        self.l298n_pwm.ChangeDutyCycle(self.movement_speed)  # Reduced speed
        self.motor_moving = True
    
    def mgm_backward(self):
        """Move the metal gearmotor backward at reduced speed"""
        GPIO.output(L298N_IN1, GPIO.HIGH)
        GPIO.output(L298N_IN2, GPIO.LOW)
        self.l298n_pwm.ChangeDutyCycle(self.movement_speed)  # Reduced speed
        self.motor_moving = True
    
    def mgm_stop(self):
        """Stop the metal gearmotor"""
        GPIO.output(L298N_IN1, GPIO.LOW)
        GPIO.output(L298N_IN2, GPIO.LOW)
        self.l298n_pwm.ChangeDutyCycle(0)
        self.motor_moving = False
        self.motor_last_moved = time.time()
    
    def stepped_movement_return(self):
        """Move servo in steps while returning to start position"""
        print("Starting stepped return movement with intermittent anti-clockwise motion")
        self.current_state = "RETURNING"
        steps = 0
        
        while self.current_state == "RETURNING" and steps < self.max_steps:
            # Move anti-clockwise
            self.motor_command = "mg996r_anticlockwise"
            self.mg996r_anti()  # Directly call motor function
            time.sleep(self.step_duration)
            
            # Stop the motor
            self.motor_command = "mg996r_stop"
            self.mg996r_stop()  # Directly call stop function
            
            # Pause for detection
            time.sleep(self.pause_duration)
            
            steps += 1
            print(f"Return step {steps}/{self.max_steps} completed")
        
        if steps >= self.max_steps:
            print("Return timeout, no start QR detected")
            self.motor_command = "mg996r_stop"
            self.mg996r_stop()




    def step_rotation(self, direction):
        """Rotate in small steps with pauses for better QR detection"""
        if direction == "clockwise":
            self.mg996r_clock()
            time.sleep(self.step_pause)  # Move for a short time
            self.mg996r_stop()
        elif direction == "anticlockwise":
            self.mg996r_anti()
            time.sleep(self.step_pause)  # Move for a short time
            self.mg996r_stop()
    
    def start_server(self):
        """Start the TCP server"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)
            print(f"Server started on {self.host}:{self.port}")
            
            print("Waiting for client connection...")
            self.client_socket, addr = self.server_socket.accept()
            print(f"Client connected from {addr}")
            self.is_connected = True
            
            return True
        except Exception as e:
            print(f"Server error: {e}")
            return False
    
    def setup_webcam(self):
        """Initialize the webcam"""
        try:
            self.webcam = cv2.VideoCapture(0)  # Use 0 for default USB webcam
            if not self.webcam.isOpened():
                print("Failed to open webcam")
                return False
                
            # Set resolution (optional)
            self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            print("Webcam initialized successfully")
            return True
        except Exception as e:
            print(f"Webcam setup error: {e}")
            return False
        
    # Improve frame sending reliability
    def send_data(self, data_type, data, additional_info=None):
        try:
            if data_type == "frame":
                # Validate frame before processing
                if data is None or data.size == 0:
                    print("Empty frame detected, skipping send")
                    return False

                # Encode with lower compression for reliability
                ret, buffer = cv2.imencode('.jpg', data, [
                    cv2.IMWRITE_JPEG_QUALITY, 70,  # Reduced quality for stability
                    cv2.IMWRITE_JPEG_OPTIMIZE, 1
                ])
                
                if not ret or buffer is None:
                    print("Frame encoding failed")
                    return False

                # Add header with checksum
                header = struct.pack('!I', len(buffer))
                self.client_socket.sendall(header + buffer.tobytes())
                return True
               
            elif data_type == "screenshot":
                # Encode screenshot as JPEG
                ret, buffer = cv2.imencode('.jpg', data, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not ret:
                    print("Failed to encode screenshot")
                    return False
                
                # Create JSON header with metadata
                header = {
                    "type": "screenshot",
                    "size": len(buffer),
                    "qr_type": additional_info.get("qr_type", "unknown"),
                    "qr_text": additional_info.get("qr_text", "unknown"),
                    "timestamp": int(time.time())
                }
                
                # Convert header to bytes and send its size first
                header_bytes = json.dumps(header).encode()
                self.client_socket.sendall(struct.pack('!I', len(header_bytes)))
                
                # Send header and then image data
                self.client_socket.sendall(header_bytes)
                self.client_socket.sendall(buffer)
                
            elif data_type == "counts":
                # Send counts as JSON
                counts = {
                    "type": "counts",
                    "box_count": self.box1_count,
                    "box2_count": self.box2_count,
                    "current_state": self.current_state,
                    "current_cycle": self.current_cycle
                }
                
                # Convert to bytes and send
                counts_bytes = json.dumps(counts).encode()
                self.client_socket.sendall(struct.pack('!I', len(counts_bytes)))
                self.client_socket.sendall(counts_bytes)
                
            return True
        except Exception as e:
            print(f"Send error: {e}")
            self.is_connected = False
            return False
    
    def receive_command(self):
        """Receive command from client"""
        try:
            # Receive data size first (4 bytes)
            size_data = self.client_socket.recv(4)
            if not size_data:
                return None
            
            size = struct.unpack('!I', size_data)[0]
            
            # Receive the actual data
            data = b""
            remaining = size
            while remaining > 0:
                chunk = self.client_socket.recv(min(4096, remaining))
                if not chunk:
                    return None
                data += chunk
                remaining -= len(chunk)
            
            # Parse command
            command = json.loads(data.decode())
            return command
        except Exception as e:
            print(f"Receive error: {e}")
            self.is_connected = False
            return None
    
        # In QRServerPi class
    def stream_video(self):
        """Stream video with frame validation"""
        fps_count = 0
        fps_start = time.time()
        
        while self.is_streaming and self.is_connected:
            # Capture frame
            ret, frame = self.webcam.read()
            if not ret or frame is None:
                print("Webcam read failed, reinitializing...")
                self.webcam.release()
                time.sleep(1)
                self.setup_webcam()
                continue

            # Validate frame dimensions
            if frame.shape[0] < 100 or frame.shape[1] < 100:
                print("Invalid frame dimensions")
                continue

            # Send frame to client
            if not self.send_data("frame", frame):
                break
            # Calculate and print FPS every second
            fps_count += 1
            if time.time() - fps_start >= 1.0:
                fps = fps_count / (time.time() - fps_start)
                print(f"Streaming at {fps:.2f} FPS")
                fps_count = 0
                fps_start = time.time()
                
                # Send current counts to client
                self.send_data("counts", None)
            
            # Adjust delay based on motor state
            # In QRServerPi.stream_video method
            if self.motor_moving:
                # Lower resolution during movement
                ret, small_frame = self.webcam.read()

            else:
                # Full resolution when stationary
                ret, frame = self.webcam.read()
                if ret:
                    self.send_data("frame", frame)
    
    def motor_control_thread(self):
        """Control motors based on current state and QR detection with step-wise movement"""
        last_step_time = 0
        step_interval = self.step_pause + self.scan_pause_time
        
        while not self.should_stop:
            current_time = time.time()
            
            # Step-wise movement mode
            if self.step_mode and self.current_state not in ["WAITING_FOR_START", "CYCLE_COMPLETE", "COMPLETE"]:
                if self.motor_command == "mg996r_clockwise" and current_time - last_step_time > step_interval:
                    self.step_rotation("clockwise")
                    last_step_time = current_time
                elif self.motor_command == "mg996r_anticlockwise" and current_time - last_step_time > step_interval:
                    self.step_rotation("anticlockwise")
                    last_step_time = current_time
            # Direct motor control mode
            else:
                if self.motor_command == "mg996r_clockwise":
                    self.mg996r_clock()
                elif self.motor_command == "mg996r_anticlockwise":
                    self.mg996r_anti()
                elif self.motor_command == "mg996r_stop":
                    self.mg996r_stop()
                elif self.motor_command == "gearmotor_forward":
                    self.mgm_forward()
                elif self.motor_command == "gearmotor_backward":
                    self.mgm_backward()
                elif self.motor_command == "gearmotor_stop":
                    self.mgm_stop()
            
            time.sleep(0.05)  # Short delay for motor control
    
    def take_screenshot(self, qr_type, qr_text):
        """Take a screenshot and send it to the client"""
        # Remove the motor moving check
        ret, frame = self.webcam.read()
        if ret:
            self.send_data("screenshot", frame, {"qr_type": qr_type, "qr_text": qr_text})

    def check_next_cycle(self):
        """Check if a new cycle should start or if process is complete"""
        if self.current_state == "CYCLE_COMPLETE":
            # Reset to waiting for start with incremented cycle number
            self.current_state = "WAITING_FOR_START"
            print(f"Ready for cycle {self.current_cycle}")
            
            # Report current counts
            print(f"Current counts: Box1={self.box1_count}, Box2={self.box2_count}")
            
            # Send updated counts to client
            self.send_data("counts", None)
    
    def run_client_detection(self):
        """Run detection on client side (receive and process results)"""
        while self.is_connected:
            # Receive detection results from client
            command = self.receive_command()
            if not command:
                print("Client disconnected")
                self.is_connected = False
                break
            
            if command.get("type") == "qr_detection":
                # Process QR detection
                qr_text = command.get("qr_text")
                if qr_text:
                    self.process_qr_detection(qr_text)
            elif command.get("type") == "control":
                # Process control command
                control_cmd = command.get("command")
                if control_cmd == "step_mode_on":
                    self.step_mode = True
                    print("Step mode enabled")
                elif control_cmd == "step_mode_off":
                    self.step_mode = False
                    print("Step mode disabled")
                elif control_cmd == "reset":
                    self.reset_state()
            
            time.sleep(0.01)  # Small delay to avoid CPU overload
    
    def reset_state(self):
        """Reset the state machine and counters"""
        self.current_state = "WAITING_FOR_START"
        self.box1_count = 0
        self.box2_count = 0
        self.current_cycle = 1
        self.motor_command = "mg996r_stop"
        print("System reset complete")
        
        # Send updated counts to client
        self.send_data("counts", None)
    
    def run(self):
        """Main execution function"""
        try:
            # Setup hardware
            self.setup_gpio()
            
            # Setup webcam
            if not self.setup_webcam():
                print("Failed to setup webcam. Exiting.")
                return
            
            # Start TCP server
            if not self.start_server():
                print("Failed to start server. Exiting.")
                return
            
            # Start motor control thread
            self.motor_thread = threading.Thread(target=self.motor_control_thread)
            self.motor_thread.daemon = True
            self.motor_thread.start()
            
            # Start video streaming
            self.is_streaming = True
            self.stream_thread = threading.Thread(target=self.stream_video)
            self.stream_thread.daemon = True
            self.stream_thread.start()
            
            # Run client detection loop
            self.run_client_detection()
            
        except KeyboardInterrupt:
            print("Server terminated by user")
        except Exception as e:
            print(f"Server error: {e}")
        finally:
            # Cleanup
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up resources")
        
        # Stop streaming
        self.is_streaming = False
        if self.stream_thread:
            self.stream_thread.join(timeout=1.0)
        
        # Stop motor control
        self.should_stop = True
        if self.motor_thread:
            self.motor_thread.join(timeout=1.0)
        
        # Stop motors
        if hasattr(self, 'mg996r_pwm'):
            self.mg996r_pwm.stop()
        if hasattr(self, 'l298n_pwm'):
            self.l298n_pwm.stop()
        
        # Release webcam
        if self.webcam:
            self.webcam.release()
        
        # Close sockets
        if self.client_socket:
            self.client_socket.close()
        if self.server_socket:
            self.server_socket.close()
        
        # Cleanup GPIO
        GPIO.cleanup()
        
        print("Cleanup complete")

if __name__ == "__main__":
    server = QRServerPi()
    server.run()