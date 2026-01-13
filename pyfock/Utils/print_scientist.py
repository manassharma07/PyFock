"""
Scientist Profile Display Module

This module provides functions to display scientist profiles with ASCII art banners,
portraits, and inspirational quotes in an aesthetically formatted terminal interface.

Functions:
    get_stylish_name(name): Returns ASCII art banner for scientist name
    print_aesthetic_quote(quote, author): Prints formatted quote in decorative box
    display_scientist_profile(name): Displays complete scientist profile
    get_random_scientist(): Returns randomly selected scientist name
    main(): Main execution function

Examples:
    >>> display_scientist_profile("bohr")
    # Displays Bohr's banner, portrait, and quote
    
    >>> scientist = get_random_scientist()
    >>> display_scientist_profile(scientist)
    # Displays random scientist profile
"""

import random
import os
import time


def get_stylish_name(name):
    """
    Return ASCII art banner for a scientist's name.
    
    Provides pre-generated ASCII art banners for supported scientist names.
    If the name is not found in the banner dictionary, returns the name in uppercase.
    
    Args:
        name (str): Lowercase scientist name (e.g., 'dalton', 'bohr', 'rutherford')
    
    Returns:
        str: Multi-line ASCII art string representing the name, or uppercase name if not found
    
    Examples:
        >>> banner = get_stylish_name('bohr')
        >>> print(banner)
         ____       _          
        | __ )  ___| |__  _ __ 
        |  _ \ / _ \ '_ \| '__|
        | |_) | (_) | | | | |   
        |____/ \___/|_| |_|_|
        
        >>> banner = get_stylish_name('unknown')
        >>> print(banner)
        UNKNOWN
    
    Note:
        Supported scientists: dalton, bohr, rutherford, dirac, schrodinger
    """
    banners = {
        "dalton":  " ____        _ _              \n|  _ \  __ _| | |_ ___  _ __  \n| | | |/ _` | | __/ _ \| '_ \ \n| |_| | (_| | | || (_) | | | |\n|____/ \__,_|_|\__\___/|_| |_|",
        "bohr":    " ____       _          \n| __ )  ___| |__  _ __ \n|  _ \ / _ \ '_ \| '__|\n| |_) | (_) | | | | |   \n|____/ \___/|_| |_|_|   ",
        "rutherford": " ____        _   _               __               _ \n|  _ \ _   _| |_| |__   ___ _ __ / _| ___  _ __ __| |\n| |_) | | | | __| '_ \ / _ \ '__| |_ / _ \| '__/ _` |\n|  _ <| |_| | |_| | | |  __/ |  |  _| (_) | | | (_| |\n|_| \_\\\\__,_|\__|_| |_|\___|_|  |_|  \___/|_|  \__,_|",
        "dirac":   " ____  _                  \n|  _ \(_)_ __ __ _  ___   \n| | | | | '__/ _` |/ __|  \n| |_| | | | | (_| | (__   \n|____/|_|_|  \__,_|\___|  ",
        "schrodinger": "  ____       _                     _ _                       \n / ___|  ___| |__  _ __ ___   __| (_)_ __   __ _  ___ _ __ \n \___ \ / __| '_ \| '__/ _ \ / _` | | '_ \ / _` |/ _ \ '__|\n  ___) | (__| | | | | | (_) | (_| | | | | | (_| |  __/ |   \n |____/ \___|_| |_|_|  \___/ \__,_|_|_| |_|\__, |\___|_|   \n                                           |___/            ",
    }
    return banners.get(name, name.upper())


def print_aesthetic_quote(quote, author):
    """
    Print a quote in a decorative boxed format with centered text.
    
    Creates a visually appealing bordered box using Unicode box-drawing characters
    and centers the quote text within it. The quote is word-wrapped to fit within
    a fixed width of 60 characters.
    
    Args:
        quote (str): The quote text to display
        author (str): The name of the quote's author
    
    Returns:
        None: Prints directly to stdout
    
    Examples:
        >>> print_aesthetic_quote("Science is beautiful.", "Einstein")
        
           ╔════════════════════════════════════════════════════════════╗
           ║                   Science is beautiful.                    ║
           ╠════════════════════════════════════════════════════════════╣
           ║                                              ~ Einstein    ║
           ╚════════════════════════════════════════════════════════════╝
    
    Note:
        - Box width is fixed at 60 characters
        - Long quotes are automatically word-wrapped
        - Author name is right-aligned at the bottom
    """
    width = 70
    horizontal_line = "═" * width
    
    print(f"\n   ╔{horizontal_line}╗")
    
    # Wrap text simply for display
    words = quote.split()
    line = ""
    for word in words:
        if len(line) + len(word) + 1 > width - 4:
            print(f"   ║ {line.center(width - 4)}   ║")
            line = word
        else:
            line = f"{line} {word}" if line else word
    if line:
        print(f"   ║ {line.center(width - 4)}   ║")
        
    print(f"   ╠{horizontal_line}╣")
    print(f"   ║ {f'~ {author}'.rjust(width - 4)}   ║")
    print(f"   ╚{horizontal_line}╝\n")


def display_scientist_profile(name):
    """
    Display complete scientist profile with banner, portrait, and quote.
    
    Orchestrates the display of a scientist's profile by:
    1. Printing a stylish ASCII art name banner
    2. Reading and displaying ASCII portrait from file (ascii_{name}.txt)
    3. Printing an inspirational quote in decorative format
    
    Args:
        name (str): Lowercase scientist name (e.g., 'dalton', 'bohr', 'rutherford')
    
    Returns:
        None: Prints directly to stdout
    
    Examples:
        >>> display_scientist_profile('bohr')
        ================================================================
         ____       _          
        | __ )  ___| |__  _ __ 
        |  _ \ / _ \ '_ \| '__|
        | |_) | (_) | | | | |   
        |____/ \___/|_| |_|_|
        ================================================================
        
           [ASCII portrait from ascii_bohr.txt]
        
           ╔════════════════════════════════════════════════════════════╗
           ║  An expert is a person who has made all the mistakes...   ║
           ╠════════════════════════════════════════════════════════════╣
           ║                                                 ~ bohr     ║
           ╚════════════════════════════════════════════════════════════╝
    
    Note:
        - Requires ASCII portrait file named 'ascii_{name}.txt' in current directory
        - If portrait file not found, displays error message
        - Supported scientists: dalton, bohr, rutherford, dirac, schrodinger
    
    Raises:
        FileNotFoundError: Caught internally and displays error message if portrait file missing
    """
    # Data definition
    data = {
        "dalton": "If I have succeeded better than many who surround me, it has been chiefly - nay, I may say, almost solely - by universal assiduity.",
        "bohr": "An expert is a person who has made all the mistakes that can be made in a very narrow field.",
        "rutherford": "All science is either physics or stamp collecting.",
        "dirac": "God used beautiful mathematics in creating the world.",
        "schrodinger": "The task is not so much to see what no one has yet seen, but to think what nobody has yet thought about that which everybody sees."
    }
    
    # Print Stylish Name
    print("\n" + "="*70)
    print(get_stylish_name(name))
    print("="*70 + "\n")
    portrait = ASCII_ART[name]
    # Indent the portrait slightly for looks
    print("\n".join(["   " + line for line in portrait.split('\n')]))

    # Print Quote
    quote = data.get(name, "Science is beautiful.")
    print_aesthetic_quote(quote, name)


def get_random_scientist():
    """
    Return a randomly selected scientist name from predefined list.
    
    Selects and returns one scientist name from a weighted list where
    some scientists may appear multiple times to increase selection probability.
    
    Returns:
        str: Lowercase scientist name (e.g., 'dalton', 'bohr', 'rutherford')
    
    Examples:
        >>> scientist = get_random_scientist()
        >>> print(scientist)
        bohr
        
        >>> scientist = get_random_scientist()
        >>> print(scientist)
        dalton
    
    Note:
        - List includes: dalton (appears twice), bohr, rutherford, dirac, schrodinger
        - Each call returns a new random selection
        - Uses random.choice() for uniform distribution among list items
    """
    names_list = ["dalton", "bohr", "rutherford", "dirac", "dalton", "schrodinger"]
    return random.choice(names_list)

def print_scientist():
    """
    Main execution function for scientist profile display.
    
    1. Randomly selects a scientist from the predefined list
    2. Calls display_scientist_profile() to show complete profile
    
    Returns:
        None: Prints directly to stdout
    
    Examples:
        >>> print_scientist()
        Randomly selected: bohr...
        [1 second pause]
        [Displays complete Bohr profile with banner, portrait, and quote]
    
    Note:
        - Includes 1-second delay for suspense effect
        - Can be called directly as entry point for the module
    """
    print("\n\n Time to get inspired by a great quantum chemist :)\n")
    selected_name = get_random_scientist()
    display_scientist_profile(selected_name)

    
    
ASCII_ART = {
    'dalton': '''
......................................................................
...............              .,,,,,,...                    ...........
........ .             ......         ..,,.                     ......
......              .,,..                ......                    ...
....                .   ...   ....   .. ..  ....                      
.                 ....   ,. .,,,,.    ,..,   , .,                     
                 .    ..,.                .   ,                       
                .,.  .,,,.                 .. .,.                     
               ,.    ,.                          ,,:                  
               ,.   ..                           .,,                  
               ,.                                ,.                   
                     .        ..   .  ,  ..   .. ,.                   
                     ,      ... .:;+;,.  :-;,.,..:.                   
                  ,;-;,..    .,,;;:,:,...,-;,;:,,;,,                  
                .....::;;;-;.;; ,,.. :.  .-. :..,+:,                  
               :.,.,.,    ., .:,    .,    ::.   :, ,                  
               ..    ,         .,....      .....  ..                  
                .,  ..,.                   .,    .,:                  
                 ..                  ,,,   :,   .,..                  
                  ,   .  .        .    .,,::,.  :;.                   
                     ,,               ..  .,.   ..                    
                    .,,,          .........,.. .,                     
                   .  .....            ....     .;:..                 
                  .,  :.,,,.    .              .:;:,;,                
                 .    , .  ,.                ..::;:,::,               
                 :    : :   ... .   .      .,,:;:;:,,.,.              
                .     ,   .   ..     .,,:;;:,,::::.    ..             
               ,      :   ,..    ...,,::,:,,,:,:::.     ,             
              ,.       ,     ,, ,..,,.   . . ,,:::.     .....         
           ...,        ,.     . ...        .,,,,,,.      ,   .....    
         ....:         ,,.      .,.         ,.,.,..      ..     .     
     .,...   ,         ,..       ,..  ..   ,    ,,        ,           
      .      ,   .,     ..        ,      ,,     ,      ,. ,           
             ,   ,       ,        ..     ,     ..      ., ,           
                          ,         ,.  ,     ,.        .,            
                          ..         , ..     .          ..           
..                         .,         ,.    .:                        
....                        .         .     .                         

''',

    'dirac': '''
          ..,.,,.,,,,,,......         ,-
       ..,,::::::,,,,,::;:,,,:,       ,-
      .,:::::,,,....,,,,::,.,::,.     ,-
    ,:;::,:;:,::,,,,,,,,,..,:-:;;     .-
  .,:;::;:,,,::;:,,,;,,:,..,:;;::;    .+
 :,:::::;---::;-+::::+::;..,:--::;,   .%
,::::::::;;,,...;-;:;;+;:..;::.,--:.. .?
,:::::::,.       .,,::::,..,.   .;;, .,*
:::::;;,                          :+, ,+
::::-;                             ,-,:*
;;:;-.                              :--?
;;:;:                                ;+?
::--,                                .+*
;-+:.                                 -?
:;:,                                  ;*
:;::,.                                -*
;;;;;,      .,;;:-;:,.           .,,, ;*
::::;       .. .:-??*?;.     .;+**;,, ,*
--;;-        .:-+++--;;,    ,-**+;;: .;*
.,;:-.      .,,;??;,;,,.   .++;*%*:;..:+
. ,;;.          .. ...      ,,.,,.   ,,+
:;                ..        ....     ,,-
..                           ,       ,,-
                              .      ,.-
                     .        .      ..-
.                    .,:-;:,:::.    ...+
.,;:,                 ,::,,;-;.     , ,+
 .:.:             .,,,;-;-++--::.   , ,-
,.:  ,            :;:,,::,,,,,,::  ,. ,-
. ,  .,           ,:::::::::,,,.   ,  ,-
 .:,  ,:.              ..,,,,.    ,.  ,-
  ,..  .:,.            ,;;;;,    ..   ,-
  : ,    ,,,.                   ,.    ,-
  ,. .    ......               ,....  ,-
  .,  ,.   ....::,.       .  .,.   ..,:-
   .,   .  .......,:::::,::::,        ,-
    ,    ..  ..........:,,            ,+

''',

    'schrodinger': '''
                               :-----;.                               
                      ::::;;;;;;;;;;;;;;;;;:::;,                      
                    .,;;;;:::;;:,,,,,,,,,,::;;:,..                    
                 .-------;;;;;;,.,,.,;,..,;;;;;;;;;;;                 
               ,:;;;;;;;;;;:::::,... :..,:::::::::..;;                
               :----;---;;:,,,,,,..  ,...,,,,,:--;..::.               
             .;-------+,                        .;+--;--;             
              ;;;;;;;:    .                       ,;:::::,            
              ;-;;;;::    ..                       ,-;;;;;.           
              ;-----:.. .  .                         ,;---.           
              .:;;;;:,,..                            .:;;,            
               :;-----;;,                            .;+-.            
               :------;::;                           ;;;;             
               :;;;;;:,,,.                          .:::.             
               ;+-+-;;;;:.,:::-:,,.      ,-;;:,.    .--:.             
              ..;--+-,...--:;-:. .;,:: .;;:;-:. .:  .:,.              
              ,;,:-;,,;+*-,.,;. ..+*-,.+-...;,   ;+-:. ..             
              -; .+: . .,;.      .+;   .;        ;,. ;...             
              ...;;. ,,. .,     ,:,,,   .,     .,.   .                
              ..:+  ,::.   ..,,,,..:,     ..,,..      .               
               ....,:::,        .. ,.                  .              
                .  ,:,,...      :....                                 
                  ..,;;:::.     .,;*-,  .          ..,.               
                     .,,,,  .      . ..            .                  
                     .,,,: ... ,..,,. .           ..                  
                     -;;;;.    ;;;:,,,,......    ,.                   
                   .,;...,..     ..              .                    
                  .:;-  ;;,,.                    .                    
                ,;:::;  .-;::::.             ,         ,              
            .,...,,,,,   ,;-;:,,...          .         .              
          .;;;;;;;;;;;,.  .:+++-,.,,.... .  .   .      ,              
     ,,,:,::.,,,,::,,:,,.    .;:;;:....   .     .                     
 .. .,:;;:,:,,;;;;;::::.,      .::;.     ..    ..                     
           ,    ,-;;;;;;;,                 ,;-:.,          .          
           .     ..,,... ....+-,,,,...::,,,,,,.,.                     
          .,         ,   .:;;+%;;--;;;;;---;;;::,                     
          ,               .:;;?;;:;;:;:;-+;;;:;,                      
         ..                ,;::-:,,,,,,::;;;::,.                      
           ,,.              ,;::.  .         ...                      

''',

    'bohr': '''
                                                                      
                               .,   ......                            
                          ..,,..,., ,,.,.,,,,..                       
                        ,:,. ..., ,. ,.,., .,:,,...                   
                      .,,.    .,   .,,,..,,,..,.,,:                   
                    .,,,..,... .. .. .. ....,,,....,.                 
                  .,,....,,.       .       ..,,,,,,,,.                
                  ::,   ..                      .,.,,:,               
                 ;,,.....                         .,,,:,              
                 :,,:,.                            ,,:::              
                 :::,,                             .:,::.             
                 ;,,. .                             ,,:;;             
                 ,..,,.                           ..::::,             
                ..,.. ...                         ..,:::.             
                 ,.......           .       ..     ..::,              
                 .  .,,:.   .,...,:;;::,   .;+::::::,:,.              
                 .....,.    . .,:;::,...  .-:-+++;,..:;,,             
                ..:...:,     .:::;:.:.    .-;:::,,;,..::,.            
                ,,, . ,,     .   .  .      ,,      ...:.,.            
                .,.  ,             .       .:     .  .,,,             
                 ,. ,                       ..       .::              
                  ..                         ,      .,,,              
                   ,                 .      ,;:      ::.              
                    ,  .              .;:.,:-;,    ..,                
                      ...                ..,,.      .,                
                        .                .         ..,                
                        ,           ..,,,,,:;::,.  .,,                
                       .,.        .:,,,..,,..,,,.  .,                 
                       ....               ...,... .,.                 
                    ., ..  ..           .,,,,.....,                   
                  .,.   ,    ,                 .,,                    
                ...     ..      .            ..::..                   
              ...        ,.             .....,;:., ..,...             
            .     .        ,         .,,,..,;;. .,    .. .            
        .                  ..           .:::.   .,          ..        
                             ,           ;:,.   .,               .    
                              ,.        .. .:,  .,                    
                               ..      .,   ,: ..,                    
                                ,      ,....,,., ..                   
                                 ,.     .,  .:   .,                   
                                        .    .    .                   

''',

    'rutherford': '''
                                                                      
                             ........                                 
                          ....  .   .,.,,,..                          
                      ..      ....,,:,,,::,::,                        
                   .,     ....,:::,..,::,::::::,                      
                  .,   .....           .,:::;:;;..                    
                  :.....                 .......,:,                   
                 :,...                       ..:::::                  
                 ...                     ....   :::;,                 
                 ..                       ...   .::::                 
                 ..                        ...  ..;:;                 
                . .                       ....  ..,:;.                
                  .                 ,.    ...,  ...,:.                
                 ,.;-;--:.     :--+--;-;:....:, ....:                 
                 ,,.,;++-:    ,**+++;,,,,....,, ....-.                
                  . -++:.,.  .,;:;,:;,;:,.  ..: .,:,:,                
                  .   .      ...       .    ..:.,,:.:;.               
                 ..          ..             ...,,:  ,:.               
                 .            ..           .....,-, .,                
                 .      .    .  ,          .....,:,.,.                
                  .      ,,,;-;.:          .....  .,.                 
                  ,    ..::,,;++;.         ....;..,                   
                  ..   ,.,.;,,.:-;:,.     ....,+,.                    
                   : ..,:;;:,,:;;-;;+;    ....:-.                     
                   ..,+**-,,::;--+++++,  ....::;.                     
                    ,.,:.  ...........   ...::.;;,                    
                     .     .,,,.        ...;,.,:,..                   
                     .,                ..,;,.., , .,.                 
                       ..            .,,::.,:,  ,    .,..             
                       .,,     ..,,:;;:,,,::,  ..     ,......         
                    ..,. .,.::;;;;;:,,,,::,    ,      ..    .....     
                .... ..   ,  ...  .,,::,      ,        ,              
            .....   ..    ,.    .,.:,.      .:         ..             
         ...        .     .,..   .::      .,,,          ,             
      ..          ..       -;....  ,    ,,.,:            .            
                           :;:...,.....,;:,:            ..            
                 .         ,. ...     .;, ..        .,.               
                  .        .,    ,   ,-.  ,          ..               
                   .       ...   ,  ,;,  ,            .               
                           ..,   ,..:;   .             ,              
                             .. ,    :. ,             ,               
                                                                      
                                                                      
                   :;--+-:;;, ;-;;;;+;--;;---;                        
                    .......   ... . .... ....                         
                                                                      

''',

}