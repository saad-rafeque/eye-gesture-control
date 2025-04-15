from __future__ import annotations
import sys, time, cv2, mediapipe as mp, numpy as np, pyautogui
from collections import deque
from typing import Dict

CAM_INDEX = 1
CAM_WIDTH, CAM_HEIGHT = 640, 480
FPS_LIMIT = 60
CALIB_SEC_PER_POSE = 4.0
SMOOTH = 5
STABLE_FRAMES = 3
MOVE_PIXELS = 100
CURSOR_DURATION = 0.12
COOLDOWN_SECONDS = 1.0
USE_ARROW_KEYS = False
# Horizontal classifier (unchanged)
TOLERANCE = {"left":0.45, "right":0.45}
# Vertical thresholds
VERT_FRACTION = 0.35   # fraction of calibration offset
# Blink
BLINK_THRESHOLD = 0.19
BLINK_COOLDOWN_S = 0.6
# Misc
DEBUG_WINDOW = False
SPEECH = True
DOT_RADIUS = 45

try:
    import pyttsx3
    eng = pyttsx3.init(); eng.setProperty("rate",185)
    def speak(t:str):
        if SPEECH: eng.say(t); eng.runAndWait()
except Exception:
    def speak(t:str): pass

# ---------- camera & FaceMesh -------------------
cam = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW if sys.platform.startswith("win") else 0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH,CAM_WIDTH); cam.set(cv2.CAP_PROP_FRAME_HEIGHT,CAM_HEIGHT)
if not cam.isOpened(): sys.exit(f" Camera {CAM_INDEX} not open")
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True,
                             min_detection_confidence=0.5, min_tracking_confidence=0.5)

SCREEN_W,SCREEN_H = pyautogui.size(); pyautogui.moveTo(SCREEN_W//2,SCREEN_H//2)

# Landmarks
L_EYE,R_EYE=[33,133],[362,263]; L_IRIS,R_IRIS=468,473
L_TOP,L_BOTTOM=159,145; R_TOP,R_BOTTOM=386,374
np_norm=np.linalg.norm

def to_px(lm,idx): p=lm[idx]; return np.array([int(p.x*CAM_WIDTH),int(p.y*CAM_HEIGHT)])

def iris_ratio(lm):
    li,ri = to_px(lm,L_IRIS), to_px(lm,R_IRIS)
    lL,lR = to_px(lm,L_EYE[0]), to_px(lm,L_EYE[1]); rL,rR = to_px(lm,R_EYE[0]), to_px(lm,R_EYE[1])
    rx = (((li[0]-lL[0])/(lR[0]-lL[0]+1e-6)) + ((ri[0]-rL[0])/(rR[0]-rL[0]+1e-6)))/2
    lT,lB = to_px(lm,L_TOP), to_px(lm,L_BOTTOM); rT,rB = to_px(lm,R_TOP), to_px(lm,R_BOTTOM)
    ry = (((li[1]-lT[1])/(lB[1]-lT[1]+1e-6)) + ((ri[1]-rT[1])/(rB[1]-rT[1]+1e-6)))/2
    return np.array([rx,ry])

def ear(lm):
    lT,lB = to_px(lm,L_TOP), to_px(lm,L_BOTTOM); rT,rB = to_px(lm,R_TOP), to_px(lm,R_BOTTOM)
    lL,lR = to_px(lm,L_EYE[0]), to_px(lm,L_EYE[1]); rL,rR = to_px(lm,R_EYE[0]), to_px(lm,R_EYE[1])
    def _ear(T,B,L,R): return np_norm(T-B)/(np_norm(L-R)+1e-6)
    return (_ear(lT,lB,lL,lR)+_ear(rT,rB,rL,rR))/2

CALIB_WIN="Calibrate"; cv2.namedWindow(CALIB_WIN,cv2.WINDOW_NORMAL); cv2.setWindowProperty(CALIB_WIN,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
blank=np.zeros((SCREEN_H,SCREEN_W,3),dtype=np.uint8)

def dot(pt): fr=blank.copy(); cv2.circle(fr,pt,DOT_RADIUS,(255,255,255),-1); cv2.imshow(CALIB_WIN,fr)
POSES=[("up",(SCREEN_W//2,SCREEN_H//5)),("left",(SCREEN_W//5,SCREEN_H//2)),("down",(SCREEN_W//2,SCREEN_H*4//5)),("right",(SCREEN_W*4//5,SCREEN_H//2)),("centre",(SCREEN_W//2,SCREEN_H//2))]
centroids:Dict[str,np.ndarray]={}
print("🧭 Calibration …"); speak("calibration starting")
for k,pt in POSES:
    speak(f"look {k}"); buf=[]; end=time.time()+CALIB_SEC_PER_POSE
    while time.time()<end:
        dot(pt); ok,frm=cam.read(); rgb=cv2.cvtColor(frm,cv2.COLOR_BGR2RGB); res=face_mesh.process(rgb)
        if res.multi_face_landmarks: buf.append(iris_ratio(res.multi_face_landmarks[0].landmark))
        if cv2.waitKey(1)&0xFF==ord('q'): sys.exit()
    centroids[k]=np.mean(buf,axis=0)
cv2.destroyWindow(CALIB_WIN); centre=centroids['centre']
base_dist={k:np_norm(v-centre) for k,v in centroids.items() if k in ('left','right')}
# vertical thresholds
up_offset = centre[1]-centroids['up'][1]; down_offset = centroids['down'][1]-centre[1]
thr_up = max(0.01, VERT_FRACTION*up_offset); thr_down = max(0.01, VERT_FRACTION*down_offset)
print("Calibration complete"); speak("calibration complete")

sm_x,sm_y=deque(maxlen=SMOOTH),deque(maxlen=SMOOTH); last_emit=0.0; stable=0; prev='centre'; blink_cd=0.0; prev_t=time.time()
while True:
    ok,frm=cam.read(); rgb=cv2.cvtColor(frm,cv2.COLOR_BGR2RGB); res=face_mesh.process(rgb); now=time.time(); label='centre'
    if res.multi_face_landmarks:
        lm=res.multi_face_landmarks[0].landmark; vec=iris_ratio(lm); sm_x.append(vec[0]); sm_y.append(vec[1])
        med=np.array([np.median(sm_x),np.median(sm_y)])
        # --- horizontal classifier (nearest‑centroid) ---
        dists={k:np_norm(med-c) for k,c in centroids.items() if k in ('left','right')}
        best=min(dists,key=dists.get)
        if dists[best]<=TOLERANCE[best]*base_dist[best]:
            label=best
        else:
            
            dy=med[1]-centre[1]
            if dy<-thr_up: label='up'
            elif dy>thr_down: label='down'
       
        if ear(lm)<BLINK_THRESHOLD and now>blink_cd:
            pyautogui.click(); speak('click'); blink_cd=now+BLINK_COOLDOWN_S
    # stability filter
    if label==prev: stable+=1
    else: stable=1; prev=label
    if label!='centre' and stable>=STABLE_FRAMES and now-last_emit>COOLDOWN_SECONDS:
        print('➡️',label.upper()); speak(label)
        if USE_ARROW_KEYS: pyautogui.press(label)
        else:
            dx={'left':-MOVE_PIXELS,'right':MOVE_PIXELS}.get(label,0); dy={'up':-MOVE_PIXELS,'down':MOVE_PIXELS}.get(label,0)
            pyautogui.moveRel(dx,dy,duration=CURSOR_DURATION)
        last_emit=now; stable=0
    if DEBUG_WINDOW and res.multi_face_landmarks:
        dbg=frm.copy();
        for idx in L_EYE+R_EYE+[L_IRIS,R_IRIS,L_TOP,L_BOTTOM,R_TOP,R_BOTTOM]: cv2.circle(dbg,tuple(to_px(lm,idx)),2,(0,255,0),-1)
        cv2.putText(dbg,f'dy:{med[1]-centre[1]:+.3f}',(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),1)
        cv2.putText(dbg,f'dir:{label}',(10,45),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),1)
        cv2.imshow('Debug',dbg)
    if cv2.waitKey(1)&0xFF==ord('q'): break
    if FPS_LIMIT:
        slp=max(0,1/FPS_LIMIT-(time.time()-prev_t)); time.sleep(slp); prev_t=time.time()
cam.release(); cv2.destroyAllWindows()
