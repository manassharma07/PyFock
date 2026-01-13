from PIL import Image
import sys

# ASCII characters from dense to sparse (dark to light)
ASCII_CHARS = ['@', '#', 'S', '%', '?', '*', '+', '-', ';', ':', ',', '.', ' ']

def resize_image(image, new_width=40):
    """Resize image while maintaining aspect ratio"""
    width, height = image.size
    aspect_ratio = height / width
    # Characters are taller than wide, so adjust height
    # Limit height to keep it on one screen (~25 lines max)
    new_height = int(new_width * aspect_ratio * 0.55)
    # new_height = min(new_height, 25)  # Cap at 25 lines
    return image.resize((new_width, new_height))

def grayscale_image(image):
    """Convert image to grayscale"""
    return image.convert('L')

def pixels_to_ascii(image):
    """Map each pixel to an ASCII character based on brightness"""
    pixels = image.getdata()
    ascii_str = ''
    for pixel in pixels:
        # Map 0-255 brightness to ASCII_CHARS index
        ascii_str += ASCII_CHARS[pixel * len(ASCII_CHARS) // 256]
    return ascii_str

def image_to_ascii(image_path, new_width=40):
    """Convert image to ASCII art
    
    Size presets:
    - 25: tiny (icon-sized)
    - 40: small (default, fits on screen)
    - 60: medium
    - 80: large
    
    Height is automatically capped at 25 lines to fit on one screen
    """
    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"Unable to open image: {e}")
        return None
    
    # Process the image
    image = resize_image(image, new_width)
    image = grayscale_image(image)
    
    # Convert pixels to ASCII
    ascii_str = pixels_to_ascii(image)
    
    # Split into lines based on image width
    img_width = image.width
    ascii_art = '\n'.join([ascii_str[i:i+img_width] 
                           for i in range(0, len(ascii_str), img_width)])
    
    return ascii_art

def save_ascii_art(ascii_art, output_file='ascii_art.txt'):
    """Save ASCII art to a text file"""
    with open(output_file, 'w') as f:
        f.write(ascii_art)
    print(f"ASCII art saved to {output_file}")

# Example usage
if __name__ == '__main__':
    # Replace 'your_image.jpg' with your image path
    image_path = 'rutherford.png'
    
    # Generate ASCII art (40 wide, max 25 lines - fits on one screen)
    # Try 25 for tiny, 60 for more detail (may scroll on tall images)
    ascii_art = image_to_ascii(image_path, new_width=70)
    
    if ascii_art:
        # Print to console
        print(ascii_art)