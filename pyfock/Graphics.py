import numpy as np
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *
from . import Data
import sys

# Global variables
xCoord, yCoord, zCoord, atomicNumber, totalNumAtoms = [], [], [], [], 0
bondStart, bondEnd, bondLengths = [], [], []
maxDistance, selectedAtom = 0.0, -1
_mouse_dragging, _previous_mouse_position = False, None
zoom_factor = 1.0  # New zoom variable
min_zoom, max_zoom = 0.1, 5.0  # Zoom limits

def initializeGlobalVars(mol):
    global xCoord, yCoord, zCoord, atomicNumber, totalNumAtoms, maxDistance
    totalNumAtoms = mol.natoms
    xCoord = [mol.coords[i][0] for i in range(mol.natoms)]
    yCoord = [mol.coords[i][1] for i in range(mol.natoms)]
    zCoord = [mol.coords[i][2] for i in range(mol.natoms)]
    atomicNumber = [mol.Zcharges[i] for i in range(mol.natoms)]
    
    # Center molecule and calculate max distance
    com = [sum(xCoord)/totalNumAtoms, sum(yCoord)/totalNumAtoms, sum(zCoord)/totalNumAtoms]
    for i in range(totalNumAtoms):
        xCoord[i] -= com[0]; yCoord[i] -= com[1]; zCoord[i] -= com[2]
    
    maxDistance = max(np.sqrt(x**2 + y**2 + z**2) for x,y,z in zip(xCoord, yCoord, zCoord)) * 2 + 5
    calculateBonds()

def calculateBonds():
    global bondStart, bondEnd, bondLengths
    bondStart, bondEnd, bondLengths = [], [], []
    for i in range(totalNumAtoms):
        for j in range(i+1, totalNumAtoms):
            dist = np.sqrt((xCoord[i]-xCoord[j])**2 + (yCoord[i]-yCoord[j])**2 + (zCoord[i]-zCoord[j])**2)
            if dist <= (float(Data.covalentRadius[atomicNumber[i]]) + float(Data.covalentRadius[atomicNumber[j]])):
                bondStart.append(i); bondEnd.append(j); bondLengths.append(dist)

def update_projection():
    """Update the projection matrix based on current zoom factor"""
    global zoom_factor, maxDistance
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    
    # Apply zoom by adjusting the orthographic viewing volume
    view_size = (maxDistance/2) / zoom_factor
    glOrtho(-view_size, view_size, -view_size, view_size, -100, 100)
    
    glMatrixMode(GL_MODELVIEW)

def zoom_in():
    """Zoom in function"""
    global zoom_factor
    zoom_factor = min(zoom_factor * 1.2, max_zoom)
    update_projection()
    glutPostRedisplay()

def zoom_out():
    """Zoom out function"""
    global zoom_factor
    zoom_factor = max(zoom_factor / 1.2, min_zoom)
    update_projection()
    glutPostRedisplay()

def mouse_wheel(wheel, direction, x, y):
    """Handle mouse wheel for zooming"""
    if direction > 0:
        zoom_in()
    else:
        zoom_out()

def draw_sphere(xyz, radius, color):
    glPushMatrix()
    color = [c/255.0 for c in color] + [1.0]
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, color)
    glMaterialfv(GL_FRONT, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
    glMaterialf(GL_FRONT, GL_SHININESS, 30.0)
    glTranslate(*xyz)
    glutSolidSphere(radius, 20, 20)
    glPopMatrix()

def cylinder_between(x1, y1, z1, x2, y2, z2, length, radius):
    v = [x2-x1, y2-y1, z2-z1]
    axis = (1, 0, 0) if np.hypot(v[0], v[1]) < 0.001 else np.cross(v, (0, 0, 1))
    angle = -np.arctan2(np.hypot(v[0], v[1]), v[2]) * 180/np.pi
    glPushMatrix()
    glTranslate(x1, y1, z1)
    glRotate(angle, *axis)
    gluCylinder(gluNewQuadric(), radius, radius, length, 10, 10)
    glPopMatrix()

def display():
    global selectedAtom
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    # Draw atoms
    for i in range(totalNumAtoms):
        draw_sphere([xCoord[i], yCoord[i], zCoord[i]], 
                   float(Data.atomicRadius[atomicNumber[i]])/2, 
                   Data.CPKcolorRGB[atomicNumber[i]])
    
    # Draw bonds
    for i in range(len(bondStart)):
        start, end = bondStart[i], bondEnd[i]
        x1, y1, z1 = xCoord[start], yCoord[start], zCoord[start]
        x2, y2, z2 = xCoord[end], yCoord[end], zCoord[end]
        
        # First half with start atom color
        color = [c/255.0 for c in Data.CPKcolorRGB[atomicNumber[start]]] + [1.0]
        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, color)
        cylinder_between(x1, y1, z1, (x1+x2)/2, (y1+y2)/2, (z1+z2)/2, bondLengths[i]/2, 0.1)
        
        # Second half with end atom color
        color = [c/255.0 for c in Data.CPKcolorRGB[atomicNumber[end]]] + [1.0]
        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, color)
        cylinder_between((x1+x2)/2, (y1+y2)/2, (z1+z2)/2, x2, y2, z2, bondLengths[i]/2, 0.1)
    
    # Display selected atom info and zoom level
    if selectedAtom >= 0:
        glDisable(GL_LIGHTING)
        glColor3f(1, 1, 1)
        glRasterPos3f(maxDistance/3/zoom_factor, maxDistance/3/zoom_factor, 0)
        info = f"Atom {selectedAtom}: {Data.elementSymbols[atomicNumber[selectedAtom]]} ({xCoord[selectedAtom]:.3f}, {yCoord[selectedAtom]:.3f}, {zCoord[selectedAtom]:.3f})"
        for char in info:
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(char))
        glEnable(GL_LIGHTING)
    
    # Display zoom level
    glDisable(GL_LIGHTING)
    glColor3f(0.8, 0.8, 0.8)
    glRasterPos3f(-maxDistance/2.5/zoom_factor, maxDistance/2.5/zoom_factor, 0)
    zoom_info = f"Zoom: {zoom_factor:.2f}x"
    for char in zoom_info:
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, ord(char))
    glEnable(GL_LIGHTING)
    
    glutSwapBuffers()

def mouse_click(button, state, x, y):
    global _mouse_dragging, _previous_mouse_position, selectedAtom
    if button == GLUT_LEFT_BUTTON:
        if state == GLUT_DOWN:
            _mouse_dragging = True
            _previous_mouse_position = (x, y)
            
            # Check for atom selection
            viewport = glGetIntegerv(GL_VIEWPORT)
            modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
            projection = glGetDoublev(GL_PROJECTION_MATRIX)
            
            winY = viewport[3] - y
            _, _, winZ = gluUnProject(x, winY, 0.5, modelview, projection, viewport)
            
            min_dist, closest = float('inf'), -1
            for i in range(totalNumAtoms):
                winX, winY, _ = gluProject(xCoord[i], yCoord[i], zCoord[i], modelview, projection, viewport)
                dist = np.sqrt((x - winX)**2 + (y - (viewport[3] - winY))**2)
                if dist < 20 and dist < min_dist:  # 20 pixel tolerance
                    min_dist, closest = dist, i
            
            selectedAtom = closest
            print(f"Selected atom {selectedAtom}: {Data.elementSymbols[atomicNumber[selectedAtom]] if selectedAtom >= 0 else 'None'}")
            glutPostRedisplay()
        else:
            _mouse_dragging = False

def mouse_motion(x, y):
    global _previous_mouse_position
    if _mouse_dragging and _previous_mouse_position:
        dx, dy = (x - _previous_mouse_position[0])/100.0, (y - _previous_mouse_position[1])/100.0
        glRotatef(dx * 50, 0, 1, 0)
        glRotatef(dy * 50, 1, 0, 0)
        _previous_mouse_position = (x, y)
        glutPostRedisplay()

def keyboard(key, x, y):
    global zoom_factor
    if key == b'\x1b':  # Escape key
        glutLeaveMainLoop()
    elif key == b'+' or key == b'=':  # Plus key for zoom in
        zoom_in()
        print(f"Zoom: {zoom_factor:.2f}x")
    elif key == b'-':  # Minus key for zoom out
        zoom_out()
        print(f"Zoom: {zoom_factor:.2f}x")
    elif key == b'r' or key == b'R':  # Reset zoom
        zoom_factor = 1.0
        update_projection()
        glutPostRedisplay()
        print("Zoom reset to 1.0x")

def special_keys(key, x, y):
    """Handle special keys like Page Up/Down for zooming"""
    if key == GLUT_KEY_PAGE_UP:
        zoom_in()
        print(f"Zoom: {zoom_factor:.2f}x")
    elif key == GLUT_KEY_PAGE_DOWN:
        zoom_out()
        print(f"Zoom: {zoom_factor:.2f}x")

def visualize(mol, title='PyFock | CrysX - 3D Viewer'):
    initializeGlobalVars(mol)
    
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(700, 700)
    glutCreateWindow(title)
    
    glClearColor(0.1, 0.1, 0.1, 1.0)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_POSITION, [10, 10, 10, 1])
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [1, 1, 1, 1])
    
    # Set initial projection with zoom support
    update_projection()
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    
    glutDisplayFunc(display)
    glutMouseFunc(mouse_click)
    glutMotionFunc(mouse_motion)
    glutKeyboardFunc(keyboard)
    glutSpecialFunc(special_keys)
    
    # Register mouse wheel callback if available
    try:
        glutMouseWheelFunc(mouse_wheel)
    except:
        pass  # Mouse wheel not supported in older PyOpenGL versions
    
    print("Controls:")
    print("- Click atoms to see coordinates")
    print("- Drag to rotate")
    print("- Mouse wheel: Zoom in/out")
    print("- '+' or '=': Zoom in")
    print("- '-': Zoom out") 
    print("- 'R': Reset zoom")
    print("- Page Up/Down: Zoom in/out")
    print("- ESC: Exit")
    print(f"Current zoom: {zoom_factor:.2f}x")
    
    glutMainLoop()