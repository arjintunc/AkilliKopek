import face_recognition
import cv2
from gpiozero import LED, Button
from datetime import datetime
import time
import os
import json
import logging
from pathlib import Path
import numpy as np
import signal
import sys
from contextlib import contextmanager

# Configure logging with rotation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.handlers.RotatingFileHandler(
            'face_recognition.log',
            maxBytes=10485760,  # 10MB
            backupCount=5
        ),
        logging.StreamHandler()
    ]
)
class Cloud:

    ACCOUNT_SID = 'AC20250502100000000000000000000000'
    AUTH_TOKEN = 'YOUR_TWILIO_AUTH_TOKEN'
    FROM_PHONE = '+1xxxxxxxxxx'  # Twilio'dan aldığınız numara
    TO_PHONE = '+90xxxxxxxxxx'   # Sahip numarası


    KNOWN_DIR = "known_faces"
    UNKNOWN_DIR = "unknown_faces"
    os.makedirs(KNOWN_DIR, exist_ok=True)
    os.makedirs(UNKNOWN_DIR, exist_ok=True)

    client = Client(ACCOUNT_SID, AUTH_TOKEN)


    known_encodings = []
    known_names = []

    for file in os.listdir(KNOWN_DIR):
    img_path = os.path.join(KNOWN_DIR, file)
    img = face_recognition.load_image_file(img_path)
    encodings = face_recognition.face_encodings(img)
    if encodings:
        known_encodings.append(encodings[0])
        known_names.append(os.path.splitext(file)[0])

class FaceRecognitionSystem:

    """Face recognition system with improved error handling and configuration validation"""
    def __init__(self):
        self.running = False
        self.config = self.load_config()
        self.initialize_components()
        self.known_encodings = []
        self.known_names = []
        self.load_known_faces()
        self.setup_signal_handlers()
        
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
    def signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown"""
        logging.info(f"Received signal {signum}, shutting down...")
        self.running = False
        
    def load_config(self):
        
        """Load configuration from file with validation"""

        config_path = "config.json"
        
        default_config = {
            "known_faces_dir": "known_faces",
            "capture_dir": "captured_images",
            "tolerance": 0.6,
            "frame_skip": 2,
            "led_duration": 2,
            "camera_resolution": (640, 480),
            "max_faces": 10,
            "min_face_size": 100,
            "log_level": "INFO"
        }
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    # Validate configuration
                    if not self.validate_config(user_config):
                        logging.warning("Invalid configuration, using defaults")
                        return default_config
                    return {**default_config, **user_config}
            return default_config
        except Exception as e:
            logging.error(f"Error loading config: {e}")
            return default_config
            
    def validate_config(self, config):
        """Validate configuration values"""
        try:
            if 'tolerance' in config and not 0 <= config['tolerance'] <= 1:
                return False
            if 'frame_skip' in config and config['frame_skip'] < 1:
                return False
            if 'led_duration' in config and config['led_duration'] < 0:
                return False
            return True
        except Exception:
            return False

    def initialize_components(self):
        """Initialize all hardware components with retry mechanism"""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                self.openbutton = Button(2)
                self.closebutton = Button(3)
                self.known_persons_led = LED(18)
                self.unknown_persons_led = LED(17)
                
                # Initialize camera with retry
                self.video_capture = cv2.VideoCapture(0)
                if not self.video_capture.isOpened():
                    raise RuntimeError("Could not open video capture device")
                
                # Set camera resolution
                self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['camera_resolution'][0])
                self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['camera_resolution'][1])
                
                # Create necessary directories
                Path(self.config['known_faces_dir']).mkdir(exist_ok=True)
                Path(self.config['capture_dir']).mkdir(exist_ok=True)
                
                logging.info("All components initialized successfully")
                return
                
            except Exception as e:
                if attempt < max_retries - 1:
                    logging.warning(f"Initialization attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(retry_delay)
                else:
                    logging.error(f"Failed to initialize components after {max_retries} attempts: {e}")
                    raise

    def load_known_faces(self):
        """Load all known faces from the configured directory with validation"""
        try:
            known_faces_dir = Path(self.config['known_faces_dir'])
            loaded_faces = 0
            
            for image_path in known_faces_dir.glob("*.jpg"):
                if loaded_faces >= self.config['max_faces']:
                    logging.warning(f"Maximum number of faces ({self.config['max_faces']}) reached")
                    break
                    
                try:
                    image = face_recognition.load_image_file(str(image_path))
                    encodings = face_recognition.face_encodings(image)
                    
                    if encodings:
                        self.known_encodings.append(encodings[0])
                        self.known_names.append(image_path.stem)
                        loaded_faces += 1
                        logging.info(f"Loaded known face: {image_path.stem}")
                    else:
                        logging.warning(f"No face found in image: {image_path}")
                except Exception as e:
                    logging.error(f"Error processing image {image_path}: {e}")
                    continue
                    
            if not self.known_encodings:
                raise ValueError("No known faces loaded")
                
            logging.info(f"Loaded {len(self.known_encodings)} known faces")
        except Exception as e:
            logging.error(f"Error loading known faces: {e}")
            raise

    @contextmanager
    def camera_context(self):
        """Context manager for camera"""
        try:
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                raise RuntimeError("Could not open video capture device")
            yield camera
        finally:
            if camera is not None:
                camera.release()

    def capture_image(self):
        """Capture an image using the camera with error handling"""
        try:
            with self.camera_context() as camera:
                ret, frame = camera.read()
                if not ret:
                    raise RuntimeError("Could not read frame from camera")
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_path = os.path.join(self.config['capture_dir'], f"captured_image_{timestamp}.jpg")
                
                cv2.imwrite(image_path, frame)
                logging.info(f"Image captured and saved as {image_path}")
                return image_path
        except Exception as e:
            logging.error(f"Error capturing image: {e}")
            return None

    def process_frame(self, frame):
        """Process a single frame for face recognition with performance optimizations"""
        try:
            # Convert the image from BGR color to RGB color
            rgb_frame = frame[:, :, ::-1]
            
            # Find all faces in the frame
            face_locations = face_recognition.face_locations(rgb_frame)
            
            # Filter out small faces
            face_locations = [
                (top, right, bottom, left) 
                for (top, right, bottom, left) in face_locations
                if (bottom - top) >= self.config['min_face_size']
            ]
            
            if not face_locations:
                return None, frame
            
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            # Draw rectangles around faces and add labels
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(
                    self.known_encodings, 
                    face_encoding,
                    tolerance=self.config['tolerance']
                )
                
                name = "Unknown"
                color = (0, 0, 255)  # Red for unknown
                
                if True in matches:
                    first_match_index = matches.index(True)
                    name = self.known_names[first_match_index]
                    color = (0, 255, 0)  # Green for known
                
                # Draw rectangle around face
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                
                # Draw label
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
                
                return name != "Unknown", frame
                
            return None, frame
        except Exception as e:
            logging.error(f"Error processing frame: {e}")
            return None, frame

    def run(self):
        """Main loop of the face recognition system with improved error handling"""
        self.running = True
        frame_count = 0
        last_error_time = 0
        error_count = 0
        
        try:
            while self.running:
                try:
                    ret, frame = self.video_capture.read()
                    if not ret:
                        current_time = time.time()
                        if current_time - last_error_time < 60:  # Within 1 minute
                            error_count += 1
                            if error_count >= 5:  # Too many errors
                                raise RuntimeError("Too many consecutive camera read errors")
                        else:
                            error_count = 0
                        last_error_time = current_time
                        logging.error("Error reading frame from camera")
                        time.sleep(1)
                        continue
                    
                    error_count = 0  # Reset error count on successful read
                    
                    # Process every nth frame
                    frame_count += 1
                    if frame_count % self.config['frame_skip'] != 0:
                        continue
                    
                    recognition_result, processed_frame = self.process_frame(frame)
                    
                    # Show the processed frame
                    cv2.imshow('Face Recognition', processed_frame)
                    
                    if recognition_result is True:
                        logging.info("Recognized person")
                        self.known_persons_led.on()
                        self.unknown_persons_led.off()
                    elif recognition_result is False:
                        logging.info("Unknown person")
                        self.known_persons_led.off()
                        self.unknown_persons_led.on()
                    
                    time.sleep(self.config['led_duration'])
                    
                    # Reset LEDs
                    self.known_persons_led.off()
                    self.unknown_persons_led.off()
                    
                    # Check for button press
                    if self.openbutton.is_pressed:
                        logging.info("Open button pressed")
                        self.capture_image()
                    
                    # Break loop on 'q' key press
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False
                        
                except Exception as e:
                    logging.error(f"Error in main loop: {e}")
                    time.sleep(1)  # Prevent tight error loop
                    
        except Exception as e:
            logging.error(f"Fatal error in main loop: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources with improved error handling"""
        try:
            if hasattr(self, 'video_capture') and self.video_capture is not None:
                self.video_capture.release()
            cv2.destroyAllWindows()
            if hasattr(self, 'known_persons_led'):
                self.known_persons_led.off()
            if hasattr(self, 'unknown_persons_led'):
                self.unknown_persons_led.off()
            logging.info("Cleanup completed successfully")
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")



if __name__ == "__main__":
    try:
        face_system = FaceRecognitionSystem()
        face_system.run()
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1)











