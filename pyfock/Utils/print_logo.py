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
    
    print("\n" + "="*60)
    print("PyFock Python Program for Quantum Chemistry Simulations")
    print("                                        by Manas Sharma")
    print("="*60)
    
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
    
    print("\n" + "="*60)