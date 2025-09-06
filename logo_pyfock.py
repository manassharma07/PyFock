#!/usr/bin/env python3

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
    print("PyFock ASCII Logo with Gradient Colors")
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
    
    # Alternative version with different styling
    print("\nAlternative Style:")
    print("-" * 50)
    
    alt_logo = [
        "▄▄▄▄▄▄▄ ▄   ▄ ▄▄▄▄▄▄▄ ▄▄▄▄▄▄▄ ▄▄▄▄▄▄▄ ▄   ▄",
        "█       █   █ █       █       █       █   █",
        "█    ▄  █▄ ▄█ █    ▄▄▄█   ▄   █       █▄ ▄█",
        "█   █ █  █ █  █   █▄▄▄█  █ █  █     ▄▄█ █ ",
        "█   █▄█  █ █  █    ▄▄▄█  █▄█  █    █  █ █ ",
        "█       █   █ █   █   █       █    █▄▄█   █",
        "█▄▄▄▄▄▄▄█▄▄▄█ █▄▄▄█   █▄▄▄▄▄▄▄█▄▄▄▄▄▄▄█▄▄▄█"
    ]
    
    for line in alt_logo:
        colored_line = ""
        chars_per_color = len(line) // len(colors)
        
        for i, char in enumerate(line):
            if char == ' ':
                colored_line += char
            else:
                color_index = min(i // max(1, chars_per_color), len(colors) - 1)
                colored_line += colors[color_index] + char + reset
        
        print(colored_line)

def print_simple_pyfock():
    """Simple version using basic characters."""
    colors = [
        '\033[94m',   # Blue
        '\033[96m',   # Cyan
        '\033[95m',   # Magenta
        '\033[91m',   # Light red
        '\033[93m',   # Yellow
        '\033[92m',   # Green
    ]
    reset = '\033[0m'
    
    simple_logo = [
        "####### #    # ###### ####### ####### #    #",
        "#     #  #  #  #      #     # #       #   # ",
        "#######   ##   #####  #     # #       ####  ",
        "#         ##   #      #     # #       #   # ",
        "#         ##   #      ####### ####### #    #"
    ]
    
    print("\nSimple Style:")
    print("-" * 45)
    
    for line in simple_logo:
        colored_line = ""
        char_count = 0
        
        for char in line:
            if char != ' ':
                color_index = (char_count // 8) % len(colors)
                colored_line += colors[color_index] + char + reset
                char_count += 1
            else:
                colored_line += char
        
        print(colored_line)

if __name__ == "__main__":
    print_pyfock_logo()
    print_simple_pyfock()
    print("\nNote: Colors may vary depending on your terminal's color support.")