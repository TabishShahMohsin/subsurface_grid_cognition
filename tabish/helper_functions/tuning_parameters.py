import cv2
import numpy as np
import pygame
import sys
from pygame.locals import *

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 600
SLIDER_WIDTH = 300
IMAGE_HEIGHT = 500
CONTROL_PANEL_WIDTH = 350
FONT_SIZE = 20
SLIDER_HEIGHT = 30
SLIDER_HANDLE_WIDTH = 10

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

class Slider:
    def __init__(self, name, min_val, max_val, default_val, y_pos):
        self.name = name
        self.min = min_val
        self.max = max_val
        self.value = default_val
        self.y_pos = y_pos
        self.dragging = False
        self.rect = pygame.Rect(
            CONTROL_PANEL_WIDTH - SLIDER_WIDTH - 20, 
            y_pos, 
            SLIDER_WIDTH, 
            SLIDER_HEIGHT
        )
        self.handle_rect = pygame.Rect(
            self._value_to_x() - SLIDER_HANDLE_WIDTH//2, 
            y_pos, 
            SLIDER_HANDLE_WIDTH, 
            SLIDER_HEIGHT
        )
    
    def _value_to_x(self):
        return self.rect.left + int((self.value - self.min) / (self.max - self.min) * self.rect.width
    
    def _x_to_value(self, x):
        x = max(self.rect.left, min(x, self.rect.right))
        return self.min + (x - self.rect.left) / self.rect.width * (self.max - self.min)
    
    def draw(self, surface):
        # Draw slider track
        pygame.draw.rect(surface, GRAY, self.rect)
        pygame.draw.rect(surface, BLACK, self.rect, 2)
        
        # Update handle position
        self.handle_rect.x = self._value_to_x() - SLIDER_HANDLE_WIDTH//2
        
        # Draw slider handle
        pygame.draw.rect(surface, BLUE if self.dragging else GREEN, self.handle_rect)
        pygame.draw.rect(surface, BLACK, self.handle_rect, 1)
        
        # Draw slider text
        font = pygame.font.SysFont('Arial', FONT_SIZE)
        name_text = font.render(f"{self.name}:", True, BLACK)
        value_text = font.render(f"{self.value:.2f}", True, BLACK)
        
        surface.blit(name_text, (20, self.y_pos + 5))
        surface.blit(value_text, (self.rect.right + 10, self.y_pos + 5))
    
    def handle_event(self, event):
        if event.type == MOUSEBUTTONDOWN:
            if self.handle_rect.collidepoint(event.pos):
                self.dragging = True
        elif event.type == MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == MOUSEMOTION and self.dragging:
            self.value = self._x_to_value(event.pos[0])
            return True  # Value changed
        return False

class ParametricTuner:
    def __init__(self, image_path):
        # Load the image
        self.org_img = cv2.imread(image_path)
        if self.org_img is None:
            raise ValueError("Could not load image")
        
        # Resize if too large
        if self.org_img.shape[0] > IMAGE_HEIGHT:
            scale_factor = IMAGE_HEIGHT / self.org_img.shape[0]
            self.org_img = cv2.resize(self.org_img, (0, 0), fx=scale_factor, fy=scale_factor)
        
        self.processed_img = self.org_img.copy()
        
        # Create Pygame window
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption('OpenCV Parametric Tuner')
        
        # Create sliders for common OpenCV parameters
        self.sliders = [
            Slider("Blur Kernel", 1, 31, 1, 50),
            Slider("Blur Sigma", 0, 15, 0, 100),
            Slider("Canny Thresh1", 0, 255, 100, 150),
            Slider("Canny Thresh2", 0, 255, 200, 200),
            Slider("Threshold", 0, 255, 127, 250),
            Slider("Contrast", 0.1, 3.0, 1.0, 300),
            Slider("Brightness", -100, 100, 0, 350),
            Slider("Hue", -180, 180, 0, 400),
            Slider("Saturation", 0, 3, 1, 450),
            Slider("Value", 0, 3, 1, 500)
        ]
        
        # Convert OpenCV image to Pygame surface
        self.org_pygame_surface = self.cvimage_to_pygame(self.org_img)
        self.processed_pygame_surface = self.cvimage_to_pygame(self.processed_img)
        
        # Font for instructions
        self.font = pygame.font.SysFont('Arial', FONT_SIZE)
    
    def cvimage_to_pygame(self, image):
        """Convert OpenCV image to Pygame surface"""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return pygame.image.frombuffer(image.tobytes(), image.shape[1::-1], "RGB")
    
    def apply_processing(self):
        """Apply all processing steps based on current slider values"""
        img = self.org_img.copy()
        
        # Get slider values
        blur_kernel = int(self.sliders[0].value)
        blur_sigma = self.sliders[1].value
        canny_thresh1 = self.sliders[2].value
        canny_thresh2 = self.sliders[3].value
        threshold = self.sliders[4].value
        contrast = self.sliders[5].value
        brightness = self.sliders[6].value
        hue = self.sliders[7].value
        saturation = self.sliders[8].value
        value = self.sliders[9].value
        
        # Apply blur if kernel > 1
        if blur_kernel > 1:
            # Ensure kernel is odd
            blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
            img = cv2.GaussianBlur(img, (blur_kernel, blur_kernel), blur_sigma)
        
        # Apply Canny edge detection if enabled
        if canny_thresh1 > 0 or canny_thresh2 > 0:
            img = cv2.Canny(img, canny_thresh1, canny_thresh2)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Apply threshold if enabled
        if threshold > 0:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        
        # Apply contrast and brightness
        img = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
        
        # Apply HSV adjustments
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Hue adjustment (wrapping around 180)
        h = ((h.astype(np.float32) + hue) % 180
        h = h.astype(np.uint8)
        
        # Saturation and Value adjustment
        s = cv2.multiply(s, saturation)
        v = cv2.multiply(v, value)
        
        hsv = cv2.merge([h, s, v])
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        self.processed_img = img
        self.processed_pygame_surface = self.cvimage_to_pygame(img)
    
    def run(self):
        clock = pygame.time.Clock()
        running = True
        
        while running:
            self.screen.fill(WHITE)
            
            # Handle events
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                
                # Check if any slider was changed
                slider_changed = False
                for slider in self.sliders:
                    if slider.handle_event(event):
                        slider_changed = True
                
                # If slider changed, update the processed image
                if slider_changed:
                    self.apply_processing()
            
            # Draw original and processed images
            self.screen.blit(self.org_pygame_surface, (20, 50))
            self.screen.blit(self.processed_pygame_surface, (20 + self.org_img.shape[1] + 20, 50))
            
            # Draw labels for images
            orig_label = self.font.render("Original Image", True, BLACK)
            processed_label = self.font.render("Processed Image", True, BLACK)
            self.screen.blit(orig_label, (20, 20))
            self.screen.blit(processed_label, (20 + self.org_img.shape[1] + 20, 20))
            
            # Draw control panel background
            control_panel_rect = pygame.Rect(
                WINDOW_WIDTH - CONTROL_PANEL_WIDTH, 
                0, 
                CONTROL_PANEL_WIDTH, 
                WINDOW_HEIGHT
            )
            pygame.draw.rect(self.screen, (240, 240, 240), control_panel_rect)
            pygame.draw.rect(self.screen, BLACK, control_panel_rect, 2)
            
            # Draw control panel title
            title = self.font.render("Parameter Controls", True, BLACK)
            self.screen.blit(title, (WINDOW_WIDTH - CONTROL_PANEL_WIDTH + 20, 15))
            
            # Draw sliders
            for slider in self.sliders:
                slider.draw(self.screen)
            
            # Draw instructions
            instructions = [
                "Instructions:",
                "- Drag sliders to adjust parameters",
                "- Changes update in real-time",
                "- Press ESC or close window to exit"
            ]
            
            for i, line in enumerate(instructions):
                text = self.font.render(line, True, BLACK)
                self.screen.blit(text, (WINDOW_WIDTH - CONTROL_PANEL_WIDTH + 20, 550 + i * 25))
            
            pygame.display.flip()
            clock.tick(30)
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parametric_tuner.py <image_path>")
        sys.exit(1)
    
    tuner = ParametricTuner(sys.argv[1])
    tuner.run()