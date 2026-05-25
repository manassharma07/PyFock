import pyfock
def print_pyfock_logo():
    """Print PyFock logo with gradient colors using ANSI escape codes."""
    
    # ANSI color codes for gradient (blue to purple to pink)
    colors = [
        '\033[38;5;39m',   # Bright blue
        '\033[38;5;45m',   # Cyan blue
        '\033[38;5;51m',   # Light cyan
        '\033[38;5;87m',   # Light blue
        '\033[38;5;123m',  # Light purple
        '\033[38;5;159m',  # Very light blue
        '\033[38;5;195m',  # Light cyan-white
        '\033[38;5;225m',  # Light pink
        '\033[38;5;219m',  # Pink
        '\033[38;5;213m',  # Magenta pink
    ]
    
    reset = '\033[0m'  # Reset color
    
    # ASCII art for PyFock

    logo_lines = [
        "██████  ██    ██ ███████  ██████   ██████ ██   ██",
        "██░░░██ ░██  ██░ ██░░░░░ ██░░░░██ ██░░░░░ ██  ██░ ",
        "██████   ░████░  █████   ██    ██ ██      █████░  ",
        "██░░░░    ░██░   ██░░░   ██    ██ ██      ██░░██ ",
        "██         ██    ██      ░██████░ ░██████ ██  ░██",
        "░░         ░░    ░░       ░░░░░░   ░░░░░░ ░░   ░░"
    ]
    
    # Alternative block-style logo
    block_logo = [
        "████████  ██    ██  ███████   ████████    ████████  ██    ██",
        "██     ██  ██  ██   ██       ██      ██  ██        ██  ██  ",
        "████████    ████    ██████   ██      ██  ██        █████   ",
        "██           ██     ██       ██      ██  ██        ██  ██  ",
        "██           ██     ██        ████████    ████████  ██    ██"
    ]
    
    print("\n" + "="*70)
    print("PyFock Python Program for Quantum Chemistry Simulations")
    print("                                        by Manas Sharma")
    print("="*70)
    
    # Print the logo with gradient colors
    for line in logo_lines:
        colored_line = ""
        chars_per_color = len(line) // len(colors)
        
        for i, char in enumerate(line):
            if char == ' ':
                colored_line += char
            else:
                color_index = min(i // max(1, chars_per_color), len(colors) - 1)
                colored_line += colors[color_index] + char + reset
        
        print(colored_line)
    
    print("\n" + "="*70)
    print("\nVersion: ", pyfock.__version__)
    print("Citation: M. Sharma and M. Sierka, PyFock: A Just-In-Time Compiled \nGaussian Basis DFT Python Code for CPU and GPU Architectures")
    print("\n" + "="*70)
    print("\n📧 Contact Information:")
    print("   Developer: Dr. Manas Sharma (PhD Physics)")
    print("   Email: manas.sharma@uni-jena.de")
    print("   Website: https://manas.bragitoff.com/")
    print("\n📚 Resources:")
    print("   Documentation: https://pyfock-docs.bragitoff.com")
    print("   Official Website: https://pyfock.bragitoff.com/")
    print("   GitHub Repo: https://github.com/manassharma07/pyfock")
    print("   Web GUI: https://pyfock-gui.bragitoff.com/")
    print("\n" + "="*70 + "\n")