#!/usr/bin/env python3
'''
Les Wright 21 June 2023
https://youtube.com/leslaboratory
A Python program to read, parse and display thermal data from the Topdon TC001 Thermal camera!
'''
print('Les Wright 21 June 2023')
print('https://youtube.com/leslaboratory')
print('A Python program to read, parse and display thermal data from the Topdon TC001 Thermal camera!')
print('')
print('Tested on Debian all features are working correctly')
print('This will work on the Pi However a number of workarounds are implemented!')
print('Seemingly there are bugs in the compiled version of cv2 that ships with the Pi!')
print('')
print('Key Bindings:')
print('')
print('a z: Increase/Decrease Blur')
print('s x: Floating High and Low Temp Label Threshold')
print('d c: Change Interpolated scale Note: This will not change the window size on the Pi')
print('f v: Contrast')
print('q w: Fullscreen Windowed (note going back to windowed does not seem to work on the Pi!)')
print('r t: Record and Stop')
print('p : Snapshot')
print('m : Cycle through ColorMaps')
print('h : Toggle HUD')

import cv2
import numpy as np
import argparse
import time
import io
import os
import json
from pathlib import Path



# Create directory for snapshots and thermal data
SNAPSHOT_DIR = "images/new"
THERMAL_DATA_DIR = "thermal_data/new"
Path(SNAPSHOT_DIR).mkdir(parents=True, exist_ok=True)
Path(THERMAL_DATA_DIR).mkdir(parents=True, exist_ok=True)

# Snapshot configuration
SNAPSHOT_INTERVAL = 3  # seconds
BUFFER_SIZE = 1000 # Change Later!!!!!
snapshot_buffer = []
thermal_data_buffer = snapshot_buffer
last_snapshot_time = 0

#Write JSON file for thermal data
def write_thermal_data(tmp_filename, min_temp, max_temp, avg_temp, center_temp, timestamp_str):
	data = {
	"timestamp" : timestamp_str,
	"min_temp" : min_temp,
	"max_temp" : max_temp,
	"avg_temp" : avg_temp,
	"center_temp" : center_temp,
	}
	with open(tmp_filename, 'w') as json_file:
		json.dump(data, json_file, indent=4)

def is_raspberrypi():
    try:
        with io.open('/sys/firmware/devicetree/base/model', 'r') as m:
            if 'raspberry pi' in m.read().lower(): return True
    except Exception: pass
    return False

isPi = is_raspberrypi()

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0, help="Video Device number e.g. 0, use v4l2-ctl --list-devices")
args = parser.parse_args()
    
if args.device:
    dev = args.device
else:
    dev = 0
    
#init video
cap = cv2.VideoCapture('/dev/video'+str(dev), cv2.CAP_V4L)
if isPi == True:
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0.0)
else:
    cap.set(cv2.CAP_PROP_CONVERT_RGB, False)

#256x192 General settings
width = 256 #Sensor width
height = 192 #sensor height
scale = 3 #scale multiplier
newWidth = width*scale 
newHeight = height*scale
alpha = 1.0 # Contrast control (1.0-3.0)
colormap = 0
font=cv2.FONT_HERSHEY_SIMPLEX
dispFullscreen = False
cv2.namedWindow('Thermal',cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow('Thermal', newWidth,newHeight)
rad = 0 #blur radius
threshold = 2
hud = True
recording = False
elapsed = "00:00:00"
snaptime = "None"

def rec():
    now = time.strftime("%Y%m%d--%H%M%S")
    videoOut = cv2.VideoWriter(now+'output.avi', cv2.VideoWriter_fourcc(*'XVID'),25, (newWidth,newHeight))
    return(videoOut)

def snapshot(heatmap):
    now = time.strftime("%Y%m%d-%H%M%S") 
    snaptime = time.strftime("%H:%M:%S")
    cv2.imwrite("TC001"+now+".png", heatmap)
    return snaptime
 
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        imdata,thdata = np.array_split(frame, 2)
        
        # Temperature calculations
        hi = thdata[96][128][0]
        lo = thdata[96][128][1]
        lo = lo*256
        rawtemp = hi+lo
        temp = (rawtemp/64)-273.15
        temp = round(temp,2)

        # Max temperature
        lomax = thdata[...,1].max()
        posmax = thdata[...,1].argmax()
        mcol,mrow = divmod(posmax,width)
        himax = thdata[mcol][mrow][0]
        lomax=lomax*256
        maxtemp = himax+lomax
        maxtemp = (maxtemp/64)-273.15
        maxtemp = round(maxtemp,2)

        # Min temperature
        lomin = thdata[...,1].min()
        posmin = thdata[...,1].argmin()
        lcol,lrow = divmod(posmin,width)
        himin = thdata[lcol][lrow][0]
        lomin=lomin*256
        mintemp = himin+lomin
        mintemp = (mintemp/64)-273.15
        mintemp = round(mintemp,2)

        # Average temperature
        loavg = thdata[...,1].mean()
        hiavg = thdata[...,0].mean()
        loavg=loavg*256
        avgtemp = loavg+hiavg
        avgtemp = (avgtemp/64)-273.15
        avgtemp = round(avgtemp,2)

        # Image processing
        bgr = cv2.cvtColor(imdata,  cv2.COLOR_YUV2BGR_YUYV)
        bgr = cv2.convertScaleAbs(bgr, alpha=alpha)
        bgr = cv2.resize(bgr,(newWidth,newHeight),interpolation=cv2.INTER_CUBIC)
        if rad>0:
            bgr = cv2.blur(bgr,(rad,rad))

        # Colormap application
        if colormap == 0:
            heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_JET)
            cmapText = 'Jet'
        if colormap == 1:
            heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_HOT)
            cmapText = 'Hot'
        if colormap == 2:
            heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_MAGMA)
            cmapText = 'Magma'
        if colormap == 3:
            heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_INFERNO)
            cmapText = 'Inferno'
        if colormap == 4:
            heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_PLASMA)
            cmapText = 'Plasma'
        if colormap == 5:
            heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_BONE)
            cmapText = 'Bone'
        if colormap == 6:
            heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_SPRING)
            cmapText = 'Spring'
        if colormap == 7:
            heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_AUTUMN)
            cmapText = 'Autumn'
        if colormap == 8:
            heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_VIRIDIS)
            cmapText = 'Viridis'
        if colormap == 9:
            heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_PARULA)
            cmapText = 'Parula'
        if colormap == 10:
            heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_RAINBOW)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            cmapText = 'Inv Rainbow'

        # Draw crosshairs
        cv2.line(heatmap,(int(newWidth/2),int(newHeight/2)+20),\
        (int(newWidth/2),int(newHeight/2)-20),(255,255,255),2)
        cv2.line(heatmap,(int(newWidth/2)+20,int(newHeight/2)),\
        (int(newWidth/2)-20,int(newHeight/2)),(255,255,255),2)
        cv2.line(heatmap,(int(newWidth/2),int(newHeight/2)+20),\
        (int(newWidth/2),int(newHeight/2)-20),(0,0,0),1)
        cv2.line(heatmap,(int(newWidth/2)+20,int(newHeight/2)),\
        (int(newWidth/2)-20,int(newHeight/2)),(0,0,0),1)

        # Display temperature
        cv2.putText(heatmap,str(temp)+' C', (int(newWidth/2)+10, int(newHeight/2)-10),\
        cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(heatmap,str(temp)+' C', (int(newWidth/2)+10, int(newHeight/2)-10),\
        cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0, 255, 255), 1, cv2.LINE_AA)

        if hud==True:
            # HUD display
            cv2.rectangle(heatmap, (0, 0),(160, 120), (0,0,0), -1)
            cv2.putText(heatmap,'Avg Temp: '+str(avgtemp)+' C', (10, 14),\
            cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(heatmap,'Label Threshold: '+str(threshold)+' C', (10, 28),\
            cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(heatmap,'Colormap: '+cmapText, (10, 42),\
            cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(heatmap,'Blur: '+str(rad)+' ', (10, 56),\
            cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(heatmap,'Scaling: '+str(scale)+' ', (10, 70),\
            cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(heatmap,'Contrast: '+str(alpha)+' ', (10, 84),\
            cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(heatmap,'Snapshot: '+snaptime+' ', (10, 98),\
            cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 255, 255), 1, cv2.LINE_AA)
            if recording == False:
                cv2.putText(heatmap,'Recording: '+elapsed, (10, 112),\
                cv2.FONT_HERSHEY_SIMPLEX, 0.4,(200, 200, 200), 1, cv2.LINE_AA)
            if recording == True:
                cv2.putText(heatmap,'Recording: '+elapsed, (10, 112),\
                cv2.FONT_HERSHEY_SIMPLEX, 0.4,(40, 40, 255), 1, cv2.LINE_AA)
        
        # Display floating temperatures
        if maxtemp > avgtemp+threshold:
            cv2.circle(heatmap, (mrow*scale, mcol*scale), 5, (0,0,0), 2)
            cv2.circle(heatmap, (mrow*scale, mcol*scale), 5, (0,0,255), -1)
            cv2.putText(heatmap,str(maxtemp)+' C', ((mrow*scale)+10, (mcol*scale)+5),\
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0,0,0), 2, cv2.LINE_AA)
            cv2.putText(heatmap,str(maxtemp)+' C', ((mrow*scale)+10, (mcol*scale)+5),\
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0, 255, 255), 1, cv2.LINE_AA)

        if mintemp < avgtemp-threshold:
            cv2.circle(heatmap, (lrow*scale, lcol*scale), 5, (0,0,0), 2)
            cv2.circle(heatmap, (lrow*scale, lcol*scale), 5, (255,0,0), -1)
            cv2.putText(heatmap,str(mintemp)+' C', ((lrow*scale)+10, (lcol*scale)+5),\
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0,0,0), 2, cv2.LINE_AA)
            cv2.putText(heatmap,str(mintemp)+' C', ((lrow*scale)+10, (lcol*scale)+5),\
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0, 255, 255), 1, cv2.LINE_AA)

        # Display image
        cv2.imshow('Thermal',heatmap)
        
        # Automatic snapshot functionality
        current_time = time.time()
        
        if current_time - last_snapshot_time >= SNAPSHOT_INTERVAL:
            last_snapshot_time = current_time
            
            # Generate img_filename with timestamp
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            img_filename = f"{SNAPSHOT_DIR}/snap_{timestamp}.png"
            
            # Generate img_filename with timestamp
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            thermal_data_filename = f"{THERMAL_DATA_DIR}/snap_{timestamp}.json"
            write_thermal_data(thermal_data_filename, mintemp, maxtemp, avgtemp, temp, timestamp)
            
            
            # Save snapshot
            cv2.imwrite(img_filename, heatmap)
            snapshot_buffer.append(img_filename)
            thermal_data_buffer.append(thermal_data_filename)
            snaptime = time.strftime("%H:%M:%S")
            print(f"Snapshot and thermal data taken: {img_filename}")
              
                
             

        if recording == True:
            elapsed = (time.time() - start)
            elapsed = time.strftime("%H:%M:%S", time.gmtime(elapsed)) 
            videoOut.write(heatmap)
        
        keyPress = cv2.waitKey(1)
        if keyPress == ord('a'): #Increase blur radius
            rad += 1
        if keyPress == ord('z'): #Decrease blur radius
            rad -= 1
            if rad <= 0:
                rad = 0

        if keyPress == ord('s'): #Increase threshold
            threshold += 1
        if keyPress == ord('x'): #Decrease threashold
            threshold -= 1
            if threshold <= 0:
                threshold = 0

        if keyPress == ord('d'): #Increase scale
            scale += 1
            if scale >=5:
                scale = 5
            newWidth = width*scale
            newHeight = height*scale
            if dispFullscreen == False and isPi == False:
                cv2.resizeWindow('Thermal', newWidth,newHeight)
        if keyPress == ord('c'): #Decrease scale
            scale -= 1
            if scale <= 1:
                scale = 1
            newWidth = width*scale
            newHeight = height*scale
            if dispFullscreen == False and isPi == False:
                cv2.resizeWindow('Thermal', newWidth,newHeight)

        if keyPress == ord('q'): #enable fullscreen
            dispFullscreen = True
            cv2.namedWindow('Thermal',cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('Thermal',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        if keyPress == ord('w'): #disable fullscreen
            dispFullscreen = False
            cv2.namedWindow('Thermal',cv2.WINDOW_GUI_NORMAL)
            cv2.setWindowProperty('Thermal',cv2.WND_PROP_AUTOSIZE,cv2.WINDOW_GUI_NORMAL)
            cv2.resizeWindow('Thermal', newWidth,newHeight)

        if keyPress == ord('f'): #contrast+
            alpha += 0.1
            alpha = round(alpha,1)
            if alpha >= 3.0:
                alpha=3.0
        if keyPress == ord('v'): #contrast-
            alpha -= 0.1
            alpha = round(alpha,1)
            if alpha<=0:
                alpha = 0.0

        if keyPress == ord('h'):
            if hud==True:
                hud=False
            elif hud==False:
                hud=True

        if keyPress == ord('m'): #m to cycle through color maps
            colormap += 1
            if colormap == 11:
                colormap = 0

        if keyPress == ord('r') and recording == False: #r to start reording
            videoOut = rec()
            recording = True
            start = time.time()
        if keyPress == ord('t'): #f to finish reording
            recording = False
            elapsed = "00:00:00"

        if keyPress == ord('p'): #manual snapshot
            snaptime = snapshot(heatmap)

        if keyPress == ord('q'):
            break
            cap.release()
            cv2.destroyAllWindows()
