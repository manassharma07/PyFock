# Graphics.py
# Author: Manas Sharma (manassharma07@live.com)
# This is a part of CrysX (https://bragitoff.com/crysx)
#
#
#  .d8888b.                            Y88b   d88P       8888888b.           8888888888                888      
# d88P  Y88b                            Y88b d88P        888   Y88b          888                       888      
# 888    888                             Y88o88P         888    888          888                       888      
# 888        888d888 888  888 .d8888b     Y888P          888   d88P 888  888 8888888  .d88b.   .d8888b 888  888 
# 888        888P"   888  888 88K         d888b          8888888P"  888  888 888     d88""88b d88P"    888 .88P 
# 888    888 888     888  888 "Y8888b.   d88888b  888888 888        888  888 888     888  888 888      888888K  
# Y88b  d88P 888     Y88b 888      X88  d88P Y88b        888        Y88b 888 888     Y88..88P Y88b.    888 "88b 
#  "Y8888P"  888      "Y88888  88888P' d88P   Y88b       888         "Y88888 888      "Y88P"   "Y8888P 888  888 
#                         888                                            888                                    
#                    Y8b d88P                                       Y8b d88P                                    
#                     "Y88P"                                         "Y88P"                                       
# This modeule contains some functionalities for visualization of molecules, isosurfaces (density, orbitals, etc.), 
# or even analysis (bond lengths, bond angles, etc.) 
# A big limitiation of the module is the need for OpenGL. This is not a problem for MacOS or Linux (UBUNTU)
# because there the library can be rather easily installed. However, for Windows, if I remember correctly, some
# DLLs were needed to be downloaded and installed and stuff. Will have to check it again and document it.

'''
TODO: Startegy for this module:
1. Allow to open the graphics window to visualize the molecule. (Already done)
2. Make sure that when the molecule is rotated or stuff in the graphical window then it only happens in the visualizer and the actual coordinates aren't affected.
3. It would also be good to allow a list of molecules to be visualized. That is, a list of mol objects would be passed and visualized simultaneously.
4. When multiple molecules are opened we should allow the user to select a particular molecule and change it's properties.
5. 
'''
import numpy as np
import numpy.linalg as la
import scipy
from . import Mol
from . import Data


from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *
import platform

from datetime import datetime
import tkinter as tk
from tkinter import ttk
#from tkinter import *
from tkinter.tix import *
from tkinter.filedialog import askopenfilename
import threading


# Unfortunately we are going to need some global static variables here.
# The reason being, we would need some information about the molecule like the 
# coordinates, charges, etc. But this information can't really be passed to functions like
# display() because these are just GLUT callback functions and can't be given arguments.
# https://stackoverflow.com/questions/12299295/passing-1-argument-pointer-to-glutdisplayfunc
# Hence we create some global variables now.
xCoord = []
yCoord = []
zCoord = []
atomicNumber = []
atomicSpecies = []
totalNumAtoms = 0
numAtomsSubsystem = []
maxDistance = 0.0
bondStart = []
bondEnd = []
bondLengths = []
colorIDs = []
indexDisplayList = 0
basicColors = False
_mouse_dragging = False
_previous_mouse_position = None
last_time = 0
stopGlutMainLoop = False
isOrtho = True
isPerspective = False
isGlutInitialized = False
isGlutOpen = False
width = 700
height = 700
aspectRatio = width/height
fieldOfView = 40
basicColors = False
isSelectionMode = False
eye = [0, 0, -10]
center = [0, 0, 0]
up = [0, 1, 0]
camera = eye, center, up
activeSubsystem = 0


def cursor_pos_callback(xpos, ypos):
    print(xpos, ypos)

def cylinder_between(x1, y1, z1, x2, y2, z2, height, rad):
    v = [x2-x1, y2-y1, z2-z1]
    axis = (1, 0, 0) if np.hypot(v[0], v[1]) < 0.001 else np.cross(v, (0, 0, 1))
    angle = -np.arctan2(np.hypot(v[0], v[1]), v[2])*180/np.pi

    glPushMatrix()
    glTranslate(x1, y1, z1)
    glRotate(angle, *axis)
    quadratic = gluNewQuadric()
    gluCylinder(quadratic, rad, rad, height, 20, 20)
    glPopMatrix()

def calculateBondPositions():
    global xCoord, yCoord, zCoord, atomicNumber, bondStart, bondEnd, bondLengths
    bondStart = []
    bondEnd = []
    bondLengths = []
    for i in range(totalNumAtoms):
        for j in range(i,totalNumAtoms):
            bondLength = calculateBondLength([xCoord[i],yCoord[i],zCoord[i]],[xCoord[j],yCoord[j],zCoord[j]])
            if bondLength<=(float(Data.covalentRadius[atomicNumber[i]])+float(Data.covalentRadius[atomicNumber[j]])):
                bondStart.append(i)
                bondEnd.append(j)
                bondLengths.append(bondLength)
                

def calculateBondLength(posA, posB):
    return np.sqrt(np.power(posA[0]-posB[0],2)+np.power(posA[1]-posB[1],2)+np.power(posA[2]-posB[2],2))


def draw_sphere(xyz, radius, color):
    global basicColors
    # Draws a sphere at a given position, 
    # with a given radius and color.
    glPushMatrix()
    color = [i * 1/255 for i in color]
    color.append(0.7)
    if basicColors:
        glColor4f(color[0],color[1],color[2],color[3])
        #print('test if basic was on or not before color selection')
    else:
        glMaterialfv(GL_FRONT,GL_AMBIENT_AND_DIFFUSE,color)
        #glMaterialfv(GL_FRONT,GL_DIFFUSE,color)
        no_mat = [0.0, 0.0, 0.0, 1.0]
        mat_specular = [1.0, 1.0, 1.0, 1.0]
        high_shininess = 30.0
        glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular)
        glMaterialf(GL_FRONT, GL_SHININESS, high_shininess)
        glMaterialfv(GL_FRONT, GL_EMISSION, no_mat)
        
    # print(xyz)
    glTranslate(xyz[0], xyz[1], xyz[2])
    glutSolidSphere(radius, 35, 35)
    glPopMatrix()

def genColorIDs(natoms):
    #Generates unique color IDs for the atoms
    colorIDs = []
    # print('Color IDs generated!')  
    i = 1
    for r in range(255):
        for g in range(255):
            for b in range(255):
                colorIDs.append([r,g,b])
                if i==natoms:
                    return
                i = i+1
    print(colorIDs)
    return colorIDs

def drawBonds():
    global xCoord, yCoord, zCoord, atomicNumber, bondStart, bondEnd, bondLengths, basicColors
    for i in range(len(bondStart)):
        startIndex = bondStart[i]
        endIndex = bondEnd[i]
        x1 = xCoord[startIndex]
        y1 = yCoord[startIndex]
        z1 = zCoord[startIndex]
        x2 = xCoord[endIndex]
        y2 = yCoord[endIndex]
        z2 = zCoord[endIndex]
        #Draw half the bond with the color of the first atom
        if not basicColors:
            color = Data.CPKcolorRGB[atomicNumber[startIndex]]
            color = [i * 1/255 for i in color]
            color.append(1.0)
            glMaterialfv(GL_FRONT,GL_AMBIENT_AND_DIFFUSE,color)
            #glMaterialfv(GL_FRONT,GL_DIFFUSE,color)
            no_mat = [0.0, 0.0, 0.0, 1.0]
            mat_specular = [1.0, 1.0, 1.0, 1.0]
            high_shininess = 30.0
            glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular)
            glMaterialf(GL_FRONT, GL_SHININESS, high_shininess)
            glMaterialfv(GL_FRONT, GL_EMISSION, no_mat)
            cylinder_between(x1, y1, z1, x2, y2, z2, bondLengths[i]/2.0, 0.1)
        #Draw the remaining half the bond with the color of the second atom
        if not basicColors:
            color = Data.CPKcolorRGB[atomicNumber[endIndex]]
            color = [i * 1/255 for i in color]
            color.append(1.0)
            glMaterialfv(GL_FRONT,GL_AMBIENT_AND_DIFFUSE,color)
            #glMaterialfv(GL_FRONT,GL_DIFFUSE,color)
            no_mat = [0.0, 0.0, 0.0, 1.0]
            mat_specular = [1.0, 1.0, 1.0, 1.0]
            high_shininess = 30.0
            glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular)
            glMaterialf(GL_FRONT, GL_SHININESS, high_shininess)
            glMaterialfv(GL_FRONT, GL_EMISSION, no_mat)
            cylinder_between((x2+x1)/2.0, (y2+y1)/2.0, (z2+z1)/2.0, x2, y2, z2, bondLengths[i]/2.0, 0.1)

def setOrthographicCamera():
    global maxDistance
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(-maxDistance/2, maxDistance/2, -maxDistance/2, maxDistance/2, -100.0, 100.0)
    glMatrixMode(GL_MODELVIEW)
    
def setPerspectiveCamera():
    global maxDistance, fieldOfView, aspectRatio, eye, center, up, camera
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    
    gluPerspective(fieldOfView,aspectRatio,1,100)
    
    glMatrixMode(GL_MODELVIEW)
  
def setCameraPosition(cam_distance):
    global maxDistance, eye, center, up, camera
    glLoadIdentity()
    eye, center, up = camera
    eye[2] = -cam_distance
    gluLookAt(eye[0], eye[1], eye[2],
                        center[0], center[1], center[2],
                        up[0], up[1], up[2])
    camera = eye, center, up

def display():
    global isSelectionMode, selectedAtoms, totalNumAtoms, xCoord, yCoord, zCoord, atomicNumber, colorIDs, isOrtho, isPersepective, indexDisplayList, basicColors
    
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
    glPushMatrix()
    # if isSelectionMode:
    #     #Light needs to be disabled while drawing text, to calculate it's color without the influence of light
    #     glDisable(GL_LIGHTING)
    #     glDisable(GL_LIGHT0)
    #     for i in range(len(selectedAtoms)):
    #         print('teste')
    #         glRasterPos3d(xCoord[selectedAtoms[i]]+float(atomicRadius[atomicNumber[selectedAtoms[i]]])/2+0.1, yCoord[selectedAtoms[i]], zCoord[selectedAtoms[i]])
    #         glColor4f(0.0, 0.0, 0.0, 1.0)
    #         color = [0,0,0,1]
    #         #glMaterialfv(GL_FRONT,GL_AMBIENT_AND_DIFFUSE,color)
    #         glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, ord('t'))
    #     #Light needs to be enabled after drawing text, cuz rest of the atoms and bonds should be calculated with light 
    #     #Or we could try drawing text after the atoms and bonds are drawn and then even if we disbale lights we dont need to renale them maybe
    #     glEnable(GL_LIGHTING)
    #     glEnable(GL_LIGHT0)
    if basicColors:
        #If the atoms are being drawn with basic colors then also disable the light.
        #This needed to be added here, because the text stuff required some changes to the lighting stuff.
        #So the light related statements in mouse click callback may now be redundant
        if isSelectionMode:
            glDisable(GL_LIGHTING)
            glDisable(GL_LIGHT0)
        for i in range(totalNumAtoms):
            print(colorIDs)
            draw_sphere([xCoord[i],yCoord[i],zCoord[i]],float(Data.atomicRadius[atomicNumber[i]])/2,colorIDs[i])
    else: 
        # print('here')
        for i in range(totalNumAtoms):
            draw_sphere([xCoord[i],yCoord[i],zCoord[i]],float(Data.atomicRadius[atomicNumber[i]])/2,Data.CPKcolorRGB[atomicNumber[i]])
    
    if isOrtho:
        #print('Ortho')
        setOrthographicCamera()
    elif isPerspective:
        #print('Perspective')
        setPerspectiveCamera()
    drawBonds()
    #drawOrbital()
    if indexDisplayList !=0:
        glCallList(indexDisplayList)
    glPopMatrix()
    glutSwapBuffers()
    return

def _create_rotation_matrix(angle, x, y, z):
    """ Creates a 3x3 rotation matrix. """
    if la.norm((x, y, z)) < 0.0001:
        return np.eye(3, dtype=np.float32)
    x, y, z = np.array((x, y, z))/la.norm((x, y, z))
    matrix = np.zeros((3, 3), dtype=np.float32)
    cos = np.cos(angle)
    sin = np.sin(angle)
    matrix[0, 0] = x*x*(1-cos)+cos
    matrix[1, 0] = x*y*(1-cos)+sin*z
    matrix[0, 1] = x*y*(1-cos)-sin*z
    matrix[2, 0] = x*z*(1-cos)-sin*y
    matrix[0, 2] = x*z*(1-cos)+sin*y
    matrix[1, 1] = y*y*(1-cos)+cos
    matrix[1, 2] = y*z*(1-cos)-sin*x
    matrix[2, 1] = y*z*(1-cos)+sin*x
    matrix[2, 2] = z*z*(1-cos)+cos
    return matrix


def mouseMoveCallback(x, y):
    """ Mouse move event handler for GLUT. """
    global _previous_mouse_position, camera, eye, center, up, maxDistance
    if _mouse_dragging:
        width = glutGet(GLUT_WINDOW_WIDTH)
        height = glutGet(GLUT_WINDOW_HEIGHT)
        # Set matrix mode
        glMatrixMode(GL_MODELVIEW)
        dx = (x-_previous_mouse_position[0])/width
        dy = (y-_previous_mouse_position[1])/height
        glRotatef(dx,1,0,0)
        glRotatef(dy,0,1,0)
#        rotation_intensity = la.norm((dx, dy)) * 4
#        eye, center, up = camera
#        camera_distance = la.norm(center-eye)
#        forward = (center-eye)/camera_distance
#        right = np.cross(forward, up)
#        rotation_axis = (up*dx+right*dy)
#        rotation_matrix = _create_rotation_matrix(-rotation_intensity,
#                                                  rotation_axis[0],
#                                                  rotation_axis[1],
#                                                  rotation_axis[2])
#        forward = np.dot(rotation_matrix, forward)
#        up = np.dot(rotation_matrix, up)
#        eye = center-forward*camera_distance
#        camera = eye, center, up
#        _previous_mouse_position = (x, y)
        #print(_mouse_dragging)
        # Reset matrix
        glLoadIdentity()
#        gluLookAt(eye[0], eye[1], eye[2],
#                         center[0], center[1], center[2],
#                         up[0], up[1], up[2])
#
        setOrthographicCamera()
        glutPostRedisplay()


def _mouse_move_callback(x, y):
    """ Mouse move event handler for GLUT. """
    global _previous_mouse_position, camera, eye, center, up, maxDistance, _mouse_dragging
    if _mouse_dragging:
        width = glutGet(GLUT_WINDOW_WIDTH)
        height = glutGet(GLUT_WINDOW_HEIGHT)
        dx = (x-_previous_mouse_position[0])/width
        dy = (y-_previous_mouse_position[1])/height
        glRotatef(dx,1,0,0)
        glRotatef(dy,0,1,0)
        rotation_intensity = la.norm((dx, dy)) * 4
        eye, center, up = camera
        camera_distance = la.norm(center-eye)
        forward = (center-eye)/camera_distance
        right = np.cross(forward, up)
        rotation_axis = (up*dx+right*dy)
        rotation_matrix = _create_rotation_matrix(-rotation_intensity,
                                                  rotation_axis[0],
                                                  rotation_axis[1],
                                                  rotation_axis[2])
        forward = np.dot(rotation_matrix, forward)
        up = np.dot(rotation_matrix, up)
        eye = center-forward*camera_distance
        camera = eye, center, up
        _previous_mouse_position = (x, y)
        #print(_mouse_dragging)
        # Set matrix mode
        glMatrixMode(GL_MODELVIEW)
        # Reset matrix
        glLoadIdentity()
        #eye[2] = -maxDistance
        gluLookAt(eye[0], eye[1], eye[2],
                         center[0], center[1], center[2],
                         up[0], up[1], up[2])
        glutPostRedisplay()

def colorPicking(x, y):
    global totalNumAtoms, colorIDs, selectedAtoms, selectedBonds
    #glReadBuffer(GL_COLOR_ATTACHMENT0)
    
    data = glReadPixels(x, 700-y, 1, 1, GL_RGB, GL_FLOAT)
    #glEnable(GL_LIGHTING)
    #glEnable(GL_LIGHT0)
    # print(data[0,0]*255)
    # print(colorIDs)
    for i in range(totalNumAtoms):
        if abs(data[0,0][0]*255-colorIDs[i][0])<0.0001 and abs(data[0,0][1]*255-colorIDs[i][1])<0.0001 and abs(data[0,0][2]*255-colorIDs[i][2])<0.0001:
            if i not in selectedAtoms:
                selectedAtoms.append(i)
                print('Selected atom: #'+str(i))
            else:
                selectedAtoms.remove(i)
                print('Deselected atom: #'+str(i))
    #glEnable(GL_LIGHTING)

def bondLengthButton():
    global isSelectionMode, selectedAtoms, selectedBonds
    if isSelectionMode:
        if len(selectedAtoms)<2:
            print('Please select 2 atoms to calculate their bond length!')
        elif len(selectedAtoms)==2:
            posA = [xCoord[selectedAtoms[0]], yCoord[selectedAtoms[0]], zCoord[selectedAtoms[0]]]
            posB = [xCoord[selectedAtoms[1]], yCoord[selectedAtoms[1]], zCoord[selectedAtoms[1]]]
            bL = calculateBondLength(posA, posB)
            print(bL)
        elif len(selectedAtoms)>2:
            print("Please select only two atoms for their bond length!")

def toggleSelectionMode():
    global isSelectionMode
    if isSelectionMode:
        isSelectionMode = False
        print('Selection mode off')
    else:
        isSelectionMode = True
        print('Selection mode on')
        #print(colorIDs)

def comCalculator(subsystem):
    global xCoord, yCoord, zCoord, atomicSpecies, atomicNumber
    com = [0, 0, 0]
    if(subsystem==0):
        for i in range(numAtomsSubsystem[subsystem]):
            com[0] = com[0] + xCoord[i]
            com[1] = com[1] + yCoord[i]
            com[2] = com[2] + zCoord[i]
    else:
        for i in range(numAtomsSubsystem[subsystem-1],numAtomsSubsystem[subsystem-1]+numAtomsSubsystem[subsystem]):
            com[0] = com[0] + xCoord[i]
            com[1] = com[1] + yCoord[i]
            com[2] = com[2] + zCoord[i]
            
    com[0] = com[0]/numAtomsSubsystem[subsystem]
    com[1] = com[1]/numAtomsSubsystem[subsystem]
    com[2] = com[2]/numAtomsSubsystem[subsystem]
    
    return com

def shiftMol2ScreenCenter(com, subsystem):
    global xCoord, yCoord, zCoord
    if(subsystem==0):
        for i in range(numAtomsSubsystem[subsystem]):
            xCoord[i] = xCoord[i] - com[0]
            yCoord[i] = yCoord[i] - com[1]
            zCoord[i] = zCoord[i] - com[2]
    else:
        for i in range(numAtomsSubsystem[subsystem-1],numAtomsSubsystem[subsystem-1]+numAtomsSubsystem[subsystem]):
            xCoord[i] = xCoord[i] - com[0]
            yCoord[i] = yCoord[i] - com[1]
            zCoord[i] = zCoord[i] - com[2]


def _mouse_click_callback(button, status, x, y):
    """ Mouse click event handler for GLUT. """
    global _mouse_dragging, _previous_mouse_position, MOLECULES
    global isSelectionMode, basicColors
    if button == GLUT_LEFT_BUTTON:
        if status == GLUT_UP:
            _mouse_dragging = False
            _previous_mouse_position = None
            basicColors = False
            print('basic off')
            #glEnable(GL_LIGHTING)
            #glEnable(GL_LIGHT0)
            #glutPostRedisplay()
        elif status == GLUT_DOWN:
            if isSelectionMode:
                print('basic on')
                glDisable(GL_LIGHTING)
                glDisable(GL_LIGHT0)
                basicColors = True
                display()
                #For some reason I need to display twice to ensure that the selection via colorpicking works properly
                display()
                colorPicking(x, y)
                basicColors = False
                print('basic off')
                glEnable(GL_LIGHTING)
                glEnable(GL_LIGHT0)
                display()
            _mouse_dragging = True
            print('Dragging')
            _previous_mouse_position = (x, y)
            
#Timer callback
def timer(msecs):
    global isOrtho

    #if (orthoCamera):
        #print('Ortho')
    
    glutTimerFunc(1, timer, 11)


# The idle callback
def idle():
    global last_time
    time = glutGet(GLUT_ELAPSED_TIME)
    if last_time == 0 or time >= last_time + 40:
        last_time = time
        #glutPostRedisplay()
    

# The visibility callback
def visible(vis):
    if vis == GLUT_VISIBLE:
        glutIdleFunc(idle)
    else:
        glutIdleFunc(None)
        
# Special keys callback 
def specialKeyCallback(key, x, y):
    # global stopGlutMainLoop
    # global isOrtho, isPersepective
    # Zoom In
    if key == GLUT_KEY_UP:
        zoomIn()
    #Zoom out
    if key == GLUT_KEY_DOWN:
        zoomOut()
        isOrtho = True

def translateSubsystems(i, axis, sign):
    global xCoord, yCoord, zCoord
    if(activeSubsystem==0):
        for i in range(numAtomsSubsystem[activeSubsystem]):
            if(axis=='z'):
                if(sign=='+'):
                    zCoord[i] = zCoord[i] + 0.1
                elif sign=='-':
                    zCoord[i] = zCoord[i] - 0.1
            if(axis=='y'):
                if(sign=='+'):
                    yCoord[i] = yCoord[i] + 0.1
                elif sign=='-':
                    yCoord[i] = yCoord[i] - 0.1  
            if(axis=='x'):
                if(sign=='+'):
                    xCoord[i] = xCoord[i] + 0.1
                elif sign=='-':
                    xCoord[i] = xCoord[i] - 0.1  
    else:
        for i in range(numAtomsSubsystem[activeSubsystem-1],numAtomsSubsystem[activeSubsystem-1]+numAtomsSubsystem[activeSubsystem]):
            if(axis=='z'):
                if(sign=='+'):
                    zCoord[i] = zCoord[i] + 0.1
                elif sign=='-':
                    zCoord[i] = zCoord[i] - 0.1
            if(axis=='y'):
                if(sign=='+'):
                    yCoord[i] = yCoord[i] + 0.1
                elif sign=='-':
                    yCoord[i] = yCoord[i] - 0.1  
            if(axis=='x'):
                if(sign=='+'):
                    xCoord[i] = xCoord[i] + 0.1
                elif sign=='-':
                    xCoord[i] = xCoord[i] - 0.1
    #Recalculate bonds after translation
    calculateBondPositions()

#Keyboard callback
def keyboardCallback(key, x, y):
    global activeSubsystem, xCoord, yCoord, zCoord, isOrtho, isPersepective, maxDistance, activeSubsystem, numAtomsSubsystem
    # print(key, activeSubsystem, numAtomsSubsystem)
    # Convert bytes object key to string 
    key = key.decode("utf-8")
    #key = key.lower
    if key == 'w':
        translateSubsystems(activeSubsystem,'z','+')
    if key == 's':
        translateSubsystems(activeSubsystem,'z','-')
    if key == 'd':
        translateSubsystems(activeSubsystem,'x','+')
    if key == 'a':
        translateSubsystems(activeSubsystem,'x','-')
    if key == 'e':
        translateSubsystems(activeSubsystem,'y','+')
    if key == 'q':
        translateSubsystems(activeSubsystem,'y','-')
    if key == 'p':
        isOrtho = False
        isPerspective = True
        setPerspectiveCamera()
        setCameraPosition(maxDistance)
    if key == 'o':
        isPerspective = False
        isOrtho = True
        setOrthographicCamera()
        setCameraPosition(10)
    if key == 'k':
        create('png')
    glutPostRedisplay()

def windowReshapeFunc(newWidth,  newHeight ):
    # global isOrtho
    if isOrtho:
        setOrthographicCamera()
    
    glViewport(0, 0, newWidth, newHeight)
#    glMatrixMode(GL_PROJECTION)
#    glLoadIdentity()
#    gluPerspective(40.,newWidth/newHeight,1.,40.)
#    glMatrixMode(GL_MODELVIEW)
#
    
def launchOpenGLWindow(title):
    global isOrtho, isGlutOpen, camera, indexDisplayList, isPerspective
    isOrtho = True
    print("CrysX - 3D Viewer Window is now being initialized!")
    glutInit(sys.argv)
    if(platform.system()=='Linux'):
        glutSetOption(GLUT_MULTISAMPLE, 8)
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_MULTISAMPLE)
    else:
        #glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH )
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_MULTISAMPLE)
        #glEnable(GL_MULTISAMPLE)
    
    glutInitWindowSize(700,700)
    glutInitWindowPosition(270, 0)
    glutCreateWindow(title)

    isGlutOpen = True
    #Background color
    #glClearColor(1.,1.,1.,1.)
    glClearColor(1,1,1,1.)
    glShadeModel(GL_SMOOTH)
    glEnable(GL_POLYGON_SMOOTH)
    glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)
    # glDisable(GL_CULL_FACE)
    # glEnable(GL_CULL_FACE)
    # glCullFace(GL_BACK)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    lightZeroPosition = [10.,4.,10.,1.]
    lightZeroColor = [1.0,1.0,1.0,1.0] #green tinged
    glLightfv(GL_LIGHT0, GL_POSITION, lightZeroPosition)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, lightZeroColor)
    glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, 0.1)
    glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, 0.05)
    glEnable(GL_LIGHT0)
    glEnable( GL_BLEND )
    # glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    
    glutDisplayFunc(display)
    #glMatrixMode(GL_PROJECTION)
    #gluPerspective(40.,1.,1.,40.)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0.0, 700.0, 0.0, 700.0, -40, 40.0)
    glMatrixMode(GL_MODELVIEW)
    # Set up the camera (it will be changed during mouse rotation)
    camera_distance = -4*2.5
    camera = ((0, 0, camera_distance),
                      (0, 0, 0),
                      (0, 1, 0))
    camera = np.array(camera)
    eye, center, up = camera
    
    gluLookAt(eye[0], eye[1], eye[2],
                         center[0], center[1], center[2],
                         up[0], up[1], up[2])
    glPushMatrix()
    glutMouseFunc(_mouse_click_callback)
    glutMotionFunc(_mouse_move_callback)
    #glutPassiveMotionFunc(cursor_pos_callback)
    #glutMotionFunc(mouseMoveCallback)
   # Set the callback for special function
    glutSpecialFunc(specialKeyCallback)
    glutKeyboardFunc(keyboardCallback)
    glutVisibilityFunc(visible)
    glutReshapeFunc(windowReshapeFunc)
    glutTimerFunc(1, timer, 1)
    if(platform.system()=='Linux'):
        thread1 = threading.Thread(target = glutMainLoop)
        thread1.start()
    print("CrysX - 3D Viewer Window created successfully!")

def initializeGlobalVars(mol):
    global xCoord, yCoord, zCoord, atomicSpecies, atomicNumber, totalNumAtoms, maxDistance, colorIDs, numAtomsSubsystem
    totalNumAtoms = mol.natoms
    for i in range(mol.natoms):
        xCoord.append(mol.coords[i][0])
        yCoord.append(mol.coords[i][1])
        zCoord.append(mol.coords[i][2])
        atomicNumber.append(mol.Zcharges[i])
        atomicSpecies.append(mol.atomicSpecies[i])

    maxDistance = calculateMaxDistance()
    colorIDs = genColorIDs(mol.natoms)
    print(colorIDs)
    numAtomsSubsystem.append(totalNumAtoms)
    # print(numAtomsSubsystem[0])

def calculateMaxDistance():
    global xCoord, yCoord, zCoord, totalNumAtoms, maxDistance
    maxDistance = 0.0
    # print(totalNumAtoms)
    for i in range(totalNumAtoms):
        for j in range(totalNumAtoms):
            newDistance = np.sqrt(np.power((xCoord[i]-xCoord[j]),2) + np.power((yCoord[i]-yCoord[j]),2) + np.power((zCoord[i]-zCoord[j]),2))
            if (newDistance>=maxDistance):
                maxDistance = newDistance
    # print(maxDistance)
    #Slightly increase it
    maxDistance = maxDistance + 5
    return maxDistance

def visualize(mol, angle='orthographic', cameraPos = None, width=700, height=700, fieldOfView=40, title='CrysX-3D Viewer'):
    # This function simply takes in a mol object and 
    # visualizes it using OpenGL
    # INPUT:
    # mol: the mol object to visualize
    # angle: can take two values- 'orthographic' or 'perspective'
    # cameraPos: the position of the camera (3x3 array)
    # width, height: the width and the height of the GLUT window
    # fieldOfView: exactly what it says
    # title: the title of the GLUT window
    # --------------------------------
  
    # Default camera position
    if cameraPos is None:
        eye = [0, 0, -10]
        center = [0, 0, 0]
        up = [0, 1, 0]
        cameraPos = eye, center, up
    #Initialize the global variables
    initializeGlobalVars(mol)
    # The reason to create a Tkinter window instead of just a GLUT window is to allow the closing of window on MacOs.
    # Moreover, I guess it would enable using some options like bond lengths, angles, point group symmetry etc.
    root=tk.Tk()      
    root.title('CrysX')
    tk.Button(root, text = "Selection Mode", command = lambda: toggleSelectionMode()).grid(row =1, column=0, padx = 1, pady = 10)
    tk.Button(root, text = "Bond Length", command = lambda: bondLengthButton()).grid(row =2, column=0, padx = 1, pady = 10)
    launchOpenGLWindow('CrysX - 3D Viewer')
    COM = comCalculator(0)
    print('COM: ', comCalculator(0))
    shiftMol2ScreenCenter(COM,0)
    calculateBondPositions()
    
    # isGlutOpen = True
    
    if (isGlutOpen):
        # Set matrix mode
        #glMatrixMode(GL_MODELVIEW)
        # Reset matrix
        #glLoadIdentity()
        setOrthographicCamera()
        glutPostRedisplay()

    # glutMainLoop()
    tk.mainloop()
