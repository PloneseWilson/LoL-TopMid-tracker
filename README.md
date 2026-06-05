# LoL-TopMid-tracker
A non Neural Network image observation tracker that bases on openCV     
File: only need *code.py*

# Game Settings
1. Borderless to allow windows to show
2. Minimap should be placed at Bottom Right, free to change if you can figure out its position and size

# How to use
1. open League of Legend, enter a rift game with blue/red side
2. run the python code.py, type your side (blue/red) into the terminal
3. the screen capture should fit the default layout well, you can change settings referring to *parameters.md*
4. press 0 on pop-out window to exit

# Default output format
missing / exist (1) / multiple (more than 2 players)  
For detection zone, refer to *TopMidZone.png*
 
# Known problems
1. cannot verify red/blue color champions e.g. anivia on red team will be classfied as both red & blue
