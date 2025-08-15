import tkinter as tk
import time
from tkinter import*
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import filedialog, ttk
from tkinter.filedialog import askopenfilename
import cv2
from cv2 import *
from threading import*
from matplotlib import pyplot as plt
import numpy as np
import queue
from multiprocessing import Pool,Queue
import threading
import zoomed
from scipy.spatial.distance import euclidean
# from imutils.video import WebcamVideoStream
# from imutils.video import FPS
# import argparse
# import imutils

MainScreenGUI = tk.Tk()
MainScreenGUI.config(bg ="white" )
MainScreenGUI.title( "NIOT classification application")
# MainScreenGUI.geometry("1500x850+320+128")
# MainScreenGUI.minsize(900,500)
MainScreenGUI.state('zoomed')
buffer=queue.Queue()
MainScreenGUI.resizable(True , True)
video_source=1
cap =cv2.VideoCapture(video_source)



f=700 #focal length of my webcam


Frame_Horizaontal_Top_Parent = LabelFrame(MainScreenGUI , text = "" , bg = "white" , fg ="black" ,borderwidth=2 )
Frame_Horizaontal_Top_Parent.pack(fill = BOTH , expand=1 , side = "top")

Frame_Horizaontal_bottom_Parent = LabelFrame(MainScreenGUI , text = "Output" , bg = "white" , fg ="black",borderwidth=2,relief="solid" )
Frame_Horizaontal_bottom_Parent.pack(fill = BOTH , expand=1 , side = "bottom")

Frame_Horizaontal_left_child_UR = LabelFrame(Frame_Horizaontal_Top_Parent  , bg = "white" , fg ="black",borderwidth=0)
Frame_Horizaontal_left_child_UR.pack(fill = BOTH , expand=1 , side = "right")

Frame_Horizaontal_left_child_UL = LabelFrame(Frame_Horizaontal_Top_Parent,text = "Display",width= 650, height= 450 , bg = "white" , fg ="black",relief="solid",borderwidth=2)
Frame_Horizaontal_left_child_UL.pack(fill = BOTH , expand=1 , side = "left")

Frame_Horizaontal_left_child_UR_child_top = LabelFrame(Frame_Horizaontal_left_child_UR ,text = "Functions", bg = "white" , fg ="black",borderwidth=2,relief="solid")
Frame_Horizaontal_left_child_UR_child_top.pack(fill = BOTH , expand=1 , side = "top")

Frame_Horizaontal_left_child_UR_child_bottom = LabelFrame(Frame_Horizaontal_left_child_UR ,text = "Input Parameter", bg = "white" , fg ="black",borderwidth=2,relief="solid")
Frame_Horizaontal_left_child_UR_child_bottom.pack(fill = BOTH ,side = "top")

Frame_Horizaontal_left_child_BR = LabelFrame(Frame_Horizaontal_bottom_Parent , bg = "white" , fg ="black",borderwidth=0,relief="solid")
Frame_Horizaontal_left_child_BR.pack(fill = BOTH , expand=1 , side = "right")

Frame_Horizaontal_left_child_BL = LabelFrame(Frame_Horizaontal_bottom_Parent  ,text = "", bg = "white" , fg ="black",borderwidth=0)
Frame_Horizaontal_left_child_BL.pack(fill = BOTH , expand=1 , side = "left")

Frame_Horizaontal_left_child_UL.pack_propagate(False)
Display_frames =tk.Label(Frame_Horizaontal_left_child_UL,bg="white",fg="black",borderwidth=0 ) # using label 
Display_frames.pack(fill = BOTH , expand=1,anchor='center')



def set_SliderBar1(var1):
    global value1
    var1=str(SlideBar1.get())
    value1=int(var1)
    # print("======>"+var1)
    
def set_SliderBar2(var2):
    global value2
    var2=str(SlideBar2.get())
    # print("======>"+var2)
    value2=float(var2)


# def Focal_Length_Finder(measured_distance, real_width, width_in_rf_image):
#     focal_length = (width_in_rf_image * measured_distance) / real_width
#     return focal_length

# def Distance_finder(Focal_Length, real_face_width, face_width_in_frame):
#     distance = (real_face_width * Focal_Length)/face_width_in_frame
#     return distance

# uncomment when calculating weight
C1_weight=487.6
C2_weight=297
C3_weight=13.2
C4_weight=11.2
C5_weight=7.5
C6_weight=5.05

def haar_cascade():
    global img,ret,C1_count,C2_count,C3_count,C4_count,C5_count,C6_count,Flaura_count,Flaura_1,Category_1,Category_2,Category_3,Category_4,Category_5,Category_6    
    ret, img = cap.read()

    C1_count = 0
    C2_count = 0
    C3_count = 0
    C4_count = 0
    C5_count = 0
    C6_count = 0
    Fauna_count=0
    Flora_count=0
    C1 = set()
    C2 = set()
    C3 = set()
    C4 = set()
    C5 = set()
    C6 = set()
    Fauna=set()
    Flora=set()
    font=cv2.FONT_HERSHEY_SIMPLEX
    
    nodule_cascade_C1=cv2.CascadeClassifier('D:/hindustan final project/Nodule_CLassifications/Cat_01/cat01_Classifier.xml')
    nodule_cascade_C2=cv2.CascadeClassifier('D:/hindustan final project/Nodule_CLassifications/Cat_02/cat02_Classifier.xml')
    nodule_cascade_C3=cv2.CascadeClassifier('D:/hindustan final project/Nodule_CLassifications/Cat_03/cat03_Classifier.xml')
    nodule_cascade_C4=cv2.CascadeClassifier('D:/hindustan final project/Nodule_CLassifications/Cat_04/cat04_Classifier.xml')
    nodule_cascade_C5=cv2.CascadeClassifier('D:/hindustan final project/Nodule_CLassifications/Cat_05/cat05_Classifier.xml')
    nodule_cascade_C6=cv2.CascadeClassifier('D:/hindustan final project/Nodule_CLassifications/Cat_06/cat06_Classifier.xml')
    Fauna_cascade=cv2.CascadeClassifier('D:/hindustan final project/fish dataset/_LABELED-FISHES-IN-THE-WILD/classifier/cascade.xml')
    Flora_cascade=cv2.CascadeClassifier('D:/hindustan final project/fish dataset/coral reef/classifier/cascade.xml')
    Category_1=nodule_cascade_C1.detectMultiScale(img, scaleFactor=value2, minNeighbors=value1, flags=cv2.CASCADE_SCALE_IMAGE)
    Category_2=nodule_cascade_C2.detectMultiScale(img, scaleFactor=value2, minNeighbors=value1, flags=cv2.CASCADE_SCALE_IMAGE)
    Category_3=nodule_cascade_C3.detectMultiScale(img, scaleFactor=value2, minNeighbors=value1, flags=cv2.CASCADE_SCALE_IMAGE)
    Category_4=nodule_cascade_C4.detectMultiScale(img, scaleFactor=value2, minNeighbors=value1, flags=cv2.CASCADE_SCALE_IMAGE)
    Category_5=nodule_cascade_C5.detectMultiScale(img, scaleFactor=value2, minNeighbors=value1, flags=cv2.CASCADE_SCALE_IMAGE)
    Category_6=nodule_cascade_C6.detectMultiScale(img, scaleFactor=value2, minNeighbors=value1, flags=cv2.CASCADE_SCALE_IMAGE)
    Fauna_1=Fauna_cascade.detectMultiScale(img, scaleFactor=value2, minNeighbors=value1, flags=cv2.CASCADE_SCALE_IMAGE)
    Flora_1=Flora_cascade.detectMultiScale(img, scaleFactor=value2, minNeighbors=value1, flags=cv2.CASCADE_SCALE_IMAGE)
    for(x,y,w,h) in Category_1:       
        C1_count = C1_count + 1
        img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2) 
        sensitivity = 100 
        C1.add((round((x + w)/sensitivity), round((y + h)/sensitivity))) 
        C1_count = len(C1) 
        width=(h-y)*0.26458333
        W=40
        #f=750
        #f=(width*d)/W
        d=abs((W*f)/(width))
        cv2.putText(img,'C-1,d={}mm'.format(int(d)),(x,y),font,0.75,(120,0,0),2)
        print("C-1 Template Shape - >> ",img.shape) 
        print("C-1 Template size - >> ",img.size)
        print ('No. of Detected C-1 Positive ->> ' + str(C1_count))
    #elif Category_2:
    for(z,v,b,n) in Category_2:
        C2_count = C2_count + 1
        img=cv2.rectangle(img,(z,v),(z+b,v+n),(0,0,255),2)
        sensitivity = 100
        C2.add((round((z + b)/sensitivity), round((v + n)/sensitivity)))
        C2_count = len(C2)
        width=(n-v)*0.26458333
        W=40
        #f=750
        #f=(width*d)/W
        d=abs((W*f)/(width))
        cv2.putText(img,'C-2,d={}mm'.format(int(d)),(z,v),font,0.75,(120,0,0),2)
        print("C-2 Template Shape - >> ",img.shape)
        print("C-2 Template size - >> ",img.size)
        print ('No. of Detected C-2 Positive->> ' + str(C2_count))
    #elif Category_3:
    for(q,w,e,r) in Category_3:
        C3_count = C3_count + 1
        img=cv2.rectangle(img,(q,w),(q+e,w+r),(255,0,0),2)
        sensitivity = 100
        C3.add((round((q + e)/sensitivity), round((w + r)/sensitivity)))
        C3_count = len(C3)
        width=(r-w)*0.26458333
        W=30
        #f=750
        #f=(width*d)/W
        d=abs((W*f)/(width))
        cv2.putText(img,'C-3,d={}mm'.format(int(d)),(q,w),font,0.75,(120,0,0),2)
        print("C-3 Template size - >> ",img.shape)
        print("C-3 Template Shape - >> ",img.size)
        print ('No. of Detected C-3 Positive ->> ' + str(C3_count))
    #elif Category_4:
    for(aq,aw,ae,ar) in Category_4:
        # if Category_4.all():
        C4_count = C4_count + 1
        img=cv2.rectangle(img,(aq,aw),(aq+ae,aw+ar),(0,120,0),2)
        sensitivity = 100
        C4.add((round((aq + ae)/sensitivity), round((aw + ar)/sensitivity)))
        C4_count = len(C4)
        width=(ar-aw)*0.26458333
        W=30
        #f=750
        #f=(width*d)/W
        d=abs((W*f)/(width))
        cv2.putText(img,'C-6,d={}mm'.format(int(d)),(aq,aw),font,0.75,(120,0,0),2)
        print("C-4 Template Shape - >> ",img.shape)
        print("C-4 Template size - >> ",img.size)
        print ('No. of Detected C-4 Positive->> ' + str(C4_count))
        # else:
            # print("no cat 4 found")
    #elif Category_5:
    for(bq,bw,be,br) in Category_5:
        # if Category_5.all():
            # C5_count = C5_count + 1
        img=cv2.rectangle(img,(bq,bw),(bq+be,bw+br),(120,0,0),2)
        sensitivity = 100
        cx,cy= bq+be//2,bw+br//2
        img = cv2.circle(img, (cx, cy), 2, (120,0,0), -1)
        C5.add((round((bq + be)/sensitivity), round((bw + br)/sensitivity)))
        C5_count = len(C5)
        width=(br-bw)*0.26458333
        W=30
        #f=750
        #f=(width*d)/W
        d=abs((W*f)/(width))
        cv2.putText(img,'C-5,d={}mm'.format(int(d)),(bq,bw),font,0.75,(120,0,0),2)
        print("C-5 Template Shape - >> ",img.shape)
        print("C-5 Template size - >> ",img.size)
        print ('No. of Detected C-5 Positive ->> ' + str(C5_count))
        # else:
            # print("No Cat-5 found")
    #elif Category_6:
    for(cq,cw,ce,cr) in Category_6:
        # if Category_6.all():
        C6_count = C6_count + 1
        img=cv2.rectangle(img,(cq,cw),(cq+ce,cw+cr),(0,0,120),2)
        sensitivity = 100
        cx,cy= cq+ce//2,cw+cr//2
        img = cv2.circle(img, (cx, cy), 2, (0, 0, 120), -1)
        C6.add((round((cq + ce)/sensitivity), round((cw + cr)/sensitivity)))
        C6_count = len(C6)
        print(cr,cw)
        width=(cr-cw)*0.26458333
        W=30
        #f=750
        #f=(width*d)/W
        d=abs((W*f)/(width))
        cv2.putText(img,'C-6,d={}mm'.format(int(d)),(cq,cw),font,0.75,(0,0,120),2)
        # cv2.putText(img,f'Distance:{int(d)}mm  ',(50,30),scale=2)
        print("C-6 Template Shape - >> ",img.shape)
        print("C-6 Template size - >> ",img.size)
        print ('No. of Detected C-6 Positive ->> ' + str(C6_count))
    # if len(Fauna_1)>0:
    #     for(dq,dw,de,dr) in Fauna_1:
    #         # if Category_6.all():
    #         Fauna_count = Fauna_count + 1
    #         img=cv2.rectangle(img,(dq,dw),(dq+de,dw+dr),(0,0,120),2)
    #         sensitivity = 100
    #         Fauna.add((round((dq + de)/sensitivity), round((dw + dr)/sensitivity)))
    #         Fauna_count = len(Fauna)
    #         print(dr,dw)
    #         width=(dr-dw)*0.26458333
    #         W=22
    #         #f=750
    #         #f=(width*d)/W
    #         # d=abs((W*f)/(width))
    #         cv2.putText(img,'Fauna,d={}mm'.format(int(d)),(dq,dw),font,0.75,(0,0,120),2)
    #         # cv2.putText(img,f'Distance:{int(d)}mm  ',(50,30),scale=2)
    #         print("Fauna Template Shape - >> ",img.shape)
    #         print("Fauna Template size - >> ",img.size)
    #         print ('No. of Detected Fauna Positive ->> ' + str(Fauna_count))
            
    # else:
    #     print("fauna not detected")
    if len(Flora_1)>0:
        for(eq,ew,ee,er) in Flora_1:
            # if Category_6.all():
            Flora_count = Flora_count + 1
            # img=cv2.rectangle(clrImg,(eq,ew),(eq+ee,ew+er),(0,0,120),2)
            sensitivity = 100
            Flora.add((round((eq + ee)/sensitivity), round((ew + er)/sensitivity)))
            Flora_count = len(Flora)
            # print(er,ew)
            # width=(er-ew)*0.26458333
            # W=22
            # #f=750
            # #f=(width*d)/W
            # d=abs((W*f)/(width))
            cv2.putText(img,"",(eq,ew),font,0.75,(0,0,120),2)
            # cv2.putText(img,f'Distance:{int(d)}mm  ',(50,30),scale=2)
            # print("Flora Template Shape - >> ",img.shape)
            # print("Flora Template size - >> ",img.size)
            Output.delete(1.0,END)
            Output.insert(END,'Flora detected in the display')
            # print ('No. of Detected Flora Positive ->> ' + str(Flora_count))
    else:
        Output.delete(1.0,END)
        Output.insert(END,'NO Flora detected')
    
    # print("/nNo. of Detected C-1 Positive ->> : {}".format(C1_count))
    # print("/nNo. of Detected C-2 Positive ->> : {}".format(C2_count))
    # print("/nNo. of Detected C-3 Positive ->> : {}".format(C3_count))    
    # print("/nNo. of Detected C-4 Positive ->> : {}".format(C4_count))
    # print("/nNo. of Detected C-5 Positive ->> : {}".format(C5_count))
    # print("/nNo. of Detected C-6 Positive ->> : {}".format(C6_count))
    Total_weight=C1_count*C1_weight+C2_count*C2_weight+C3_count*C3_weight+C4_count*C4_weight+C5_count*C5_weight+C6_count*C6_weight 
    Total_nodules = C1_count+C2_count+C3_count+C4_count+C5_count+C6_count+Fauna_count   
    height,width,_=img.shape
    cv2.circle(img, (width//2, height//2), 1, (0,0,0), -1)
    width_seabed=343
    distance_seabed=(width_seabed*f)/(width)  
    area=height*width*0.2645833*0.2645833
    Entry_NodulesCount.delete(0, END)
    Entry_NodulesCount.insert(0,format(Total_nodules))
    Entry_WeightsCount.delete(0,END)
    Entry_WeightsCount.insert(0,"{} gms".format(Total_weight))
    Entry_Area.delete(0,END) 
    Entry_Area.insert(0,"{} sq.millimeter".format(int(area))) 
    Entry_CAmeraDist.delete(0,END)
    Entry_CAmeraDist.insert(0,"{} mm".format(distance_seabed))
    return 


'''
#              Working Method without detection:
    
def open_file():
   global img
   f_types = [('Jpg Files', '*.jpg')]
   filename = filedialog.askopenfilename(filetypes=f_types)
   img=Image.open(filename)
   img_resized=img.resize((400,200)) # new width & height
   img=ImageTk.PhotoImage(img_resized)
   Display_frames.image = img
   Display_frames.configure(image=img)
   
'''

def haar_cascade_browser():
    global img,clrImg,C1_count,C2_count,C3_count,C4_count,C5_count,C6_count,Category_1,Category_2,Category_3,Category_4,Category_5,Category_6
    # gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #nodules = nodule_cascade.detectMultiScale(gray, 5, 5)
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(10,10))
    
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    
    l, a, b = cv2.split(lab)  # split on 3 different channels
    # equ = cv2.equalizeHist(l)
    l2 = clahe.apply(l)  # apply CLAHE to the L-channel

    lab = cv2.merge((l2,a,b))  # merge channels
    clrImg = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)  # convert from LAB to RGB
    
    
    # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(10,10))
    # cl1 = clahe.apply(equ)
    # clrImg = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    C1_count = 0
    C2_count = 0
    C3_count = 0
    C4_count = 0
    C5_count = 0
    C6_count = 0
    Fauna_count=0
    
    Flora_count=0
    C1 = set()
    C2 = set()
    C3 = set()
    C4 = set()
    C5 = set()
    C6 = set()
    Flora=set()
    Fauna=set()
    
    nodule_cascade_C1=cv2.CascadeClassifier('D:/hindustan final project/Nodule_CLassifications/Cat_01/cat01_Classifier.xml')
    nodule_cascade_C2=cv2.CascadeClassifier('D:/hindustan final project/Nodule_CLassifications/Cat_02/cat02_Classifier.xml')
    nodule_cascade_C3=cv2.CascadeClassifier('D:/hindustan final project/Nodule_CLassifications/Cat_03/cat03_Classifier.xml')
    nodule_cascade_C4=cv2.CascadeClassifier('D:/hindustan final project/Nodule_CLassifications/Cat_04/cat04_Classifier.xml')
    nodule_cascade_C5=cv2.CascadeClassifier('D:/hindustan final project/Nodule_CLassifications/Cat_05/cat05_Classifier.xml')
    nodule_cascade_C6=cv2.CascadeClassifier('D:/hindustan final project/Nodule_CLassifications/Cat_06/cat06_Classifier.xml')
    Fauna_cascade=cv2.CascadeClassifier('D:/hindustan final project/fish dataset/_LABELED-FISHES-IN-THE-WILD/classifier/cascade.xml')   
    Flora_cascade=cv2.CascadeClassifier('D:/hindustan final project/fish dataset/coral reef/classifier/cascade.xml')
    # gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    font=cv2.FONT_HERSHEY_SIMPLEX
    
    Category_1=nodule_cascade_C1.detectMultiScale(clrImg, scaleFactor=value2, minNeighbors=value1, flags=cv2.CASCADE_SCALE_IMAGE)
    Category_2=nodule_cascade_C2.detectMultiScale(clrImg, scaleFactor=value2, minNeighbors=value1, flags=cv2.CASCADE_SCALE_IMAGE)
    Category_3=nodule_cascade_C3.detectMultiScale(clrImg, scaleFactor=value2, minNeighbors=value1, flags=cv2.CASCADE_SCALE_IMAGE)
    Category_4=nodule_cascade_C4.detectMultiScale(clrImg, scaleFactor=value2, minNeighbors=value1, flags=cv2.CASCADE_SCALE_IMAGE)
    Category_5=nodule_cascade_C5.detectMultiScale(clrImg, scaleFactor=value2, minNeighbors=value1, flags=cv2.CASCADE_SCALE_IMAGE)
    Category_6=nodule_cascade_C6.detectMultiScale(clrImg, scaleFactor=value2, minNeighbors=value1, flags=cv2.CASCADE_SCALE_IMAGE)
    Fauna_1=Fauna_cascade.detectMultiScale(img, scaleFactor=value2, minNeighbors=value1, flags=cv2.CASCADE_SCALE_IMAGE)
    Flora_1=Flora_cascade.detectMultiScale(img, scaleFactor=value2, minNeighbors=value1, flags=cv2.CASCADE_SCALE_IMAGE)
    for(x,y,w,h) in Category_1:       
        C1_count = C1_count + 1
        img=cv2.rectangle(clrImg,(x,y),(x+w,y+h),(0,255,0),2) 
        sensitivity = 100 
        C1.add((round((x + w)/sensitivity), round((y + h)/sensitivity))) 
        C1_count = len(C1) 
        width=(h-y)*0.26458333
        W=66
        #f=750
        #f=(width*d)/W
        d=abs((W*f)/(width))
        cv2.putText(img,'C-1,d={}mm'.format(int(d)),(x,y),font,0.75,(120,0,0),2)
        print("C-1 Template Shape - >> ",img.shape) 
        print("C-1 Template size - >> ",img.size)
    for(z,v,b,n) in Category_2:
        C2_count = C2_count + 1
        img=cv2.rectangle(clrImg,(z,v),(z+b,v+n),(0,0,255),2)
        sensitivity = 100
        C2.add((round((z + b)/sensitivity), round((v + n)/sensitivity)))
        C2_count = len(C2)
        width=(n-v)*0.26458333
        W=42
        #f=750
        #f=(width*d)/W
        d=abs((W*f)/(width))
        cv2.putText(img,'C-2,d={}mm'.format(int(d)),(z,v),font,0.75,(120,0,0),2)
        print("C-2 Template Shape - >> ",img.shape)
        print("C-2 Template size - >> ",img.size)
    for(q,w,e,r) in Category_3:
        C3_count = C3_count + 1
        img=cv2.rectangle(clrImg,(q,w),(q+e,w+r),(255,0,0),2)
        sensitivity = 100
        C3.add((round((q + e)/sensitivity), round((w + r)/sensitivity)))
        C3_count = len(C3)
        width=(r-w)*0.26458333
        W=40
        #f=750
        #f=(width*d)/W
        d=abs((W*f)/(width))
        cv2.putText(img,'C-3,d={}mm'.format(int(d)),(q,w),font,0.75,(120,0,0),2)
        print("C-3 Template size - >> ",img.shape)
        print("C-3 Template Shape - >> ",img.size)
    for(aq,aw,ae,ar) in Category_4:
        C4_count = C4_count + 1
        img=cv2.rectangle(clrImg,(aq,aw),(aq+ae,aw+ar),(0,120,0),2)
        sensitivity = 100
        C4.add((round((aq + ae)/sensitivity), round((aw + ar)/sensitivity)))
        C4_count = len(C4)
        width=(ar-aw)*0.26458333
        W=31
        #f=750
        #f=(width*d)/W
        d=abs((W*f)/(width))
        cv2.putText(img,'C-6,d={}mm'.format(int(d)),(aq,aw),font,0.75,(120,0,0),2)
        print("C-4 Template Shape - >> ",img.shape)
        print("C-4 Template size - >> ",img.size)
    for(bq,bw,be,br) in Category_5:
        C5_count = C5_count + 1
        img=cv2.rectangle(clrImg,(bq,bw),(bq+be,bw+br),(120,0,0),2)
        sensitivity = 100
        C5.add((round((bq + be)/sensitivity), round((bw + br)/sensitivity)))
        C5_count = len(C5)
        width=(br-bw)*0.26458333
        W=27
        #f=750
        #f=(width*d)/W
        d=abs((W*f)/(width))
        cv2.putText(img,'C-5,d={}mm'.format(int(d)),(bq,bw),font,0.75,(120,0,0),2)
        print("C-5 Template Shape - >> ",img.shape)
        print("C-5 Template size - >> ",img.size)
    for(cq,cw,ce,cr) in Category_6:
        C6_count = C6_count + 1
        img=cv2.rectangle(clrImg,(cq,cw),(cq+ce,cw+cr),(0,0,120),2)
        sensitivity = 100
        C6.add((round((cq + ce)/sensitivity), round((cw + cr)/sensitivity)))
        C6_count = len(C6)
        print(cr,cw)
        width=(cr-cw)*0.26458333
        W=22
        #f=750
        #f=(width*d)/W
        d=abs((W*f)/(width))
        cv2.putText(img,'C-6,d={}mm'.format(int(d)),(cq,cw),font,0.75,(0,0,120),2)
        print("C-6 Template Shape - >> ",img.shape)
        print("C-6 Template size - >> ",img.size)
    if len(Fauna_1)>0:
        for(dq,dw,de,dr) in Fauna_1:
            # if Category_6.all():
            Fauna_count = Fauna_count + 1
            img=cv2.rectangle(clrImg,(dq,dw),(dq+de,dw+dr),(0,0,120),2)
            sensitivity = 100
            Fauna.add((round((dq + de)/sensitivity), round((dw + dr)/sensitivity)))
            Fauna_count = len(Fauna)
            print(dr,dw)
            width=((dr-dw)*0.26458333)
            W=22
            # f=750
            #f=(width*d)/W
            d=abs((W*f)/(width))
            cv2.putText(img,'Fauna,d={}mm'.format(int(d)),(dq,dw),font,0.75,(0,0,120),2)
            # cv2.putText(img,f'Distance:{int(d)}mm  ',(50,30),scale=2)
            print("Fauna Template Shape - >> ",img.shape)
            print("Fauna Template size - >> ",img.size)
    else:
        print("not detected")
    if len(Flora_1)>0:
        for(eq,ew,ee,er) in Flora_1:
            # if Category_6.all():
            Flora_count = Flora_count + 1
            # img=cv2.rectangle(clrImg,(eq,ew),(eq+ee,ew+er),(0,0,120),2)
            sensitivity = 100
            Flora.add((round((eq + ee)/sensitivity), round((ew + er)/sensitivity)))
            Flora_count = len(Flora)
            # print(er,ew)
            # width=(er-ew)*0.26458333
            # W=22
            # #f=750
            # #f=(width*d)/W
            # d=abs((W*f)/(width))
            cv2.putText(img,"",(eq,ew),font,0.75,(0,0,120),2)
            # cv2.putText(img,f'Distance:{int(d)}mm  ',(50,30),scale=2)
            # print("Flora Template Shape - >> ",img.shape)
            # print("Flora Template size - >> ",img.size)
            Output.delete(1.0,END)
            Output.insert(END,'Flora detected in the display')
    else:
        Output.delete(1.0,END)
        Output.insert(END,'NO Flora detected')
        
    print("/nNo. of Detected C-1 Positive ->> : {}".format(C1_count))
    print("/nNo. of Detected C-2 Positive ->> : {}".format(C2_count))
    print("/nNo. of Detected C-3 Positive ->> : {}".format(C3_count))    
    print("/nNo. of Detected C-4 Positive ->> : {}".format(C4_count))
    print("/nNo. of Detected C-5 Positive ->> : {}".format(C5_count))
    print("/nNo. of Detected C-6 Positive ->> : {}".format(C6_count))
    print ('No. of Detected Fauna Positive ->> ' + str(Fauna_count))
    # print ('No. of Detected Flora Positive ->> ' + str(Flora_count))
    height,width,_=clrImg.shape
    area=height*width*0.0002645833
    distance_seabed=(width*f)/height
    Total_weight=C1_count*C1_weight+C2_count*C2_weight+C3_count*C3_weight+C4_count*C4_weight+C5_count*C5_weight+C6_count*C6_weight 
    Total_nodules = C1_count+C2_count+C3_count+C4_count+C5_count+C6_count+Fauna_count
    Entry_NodulesCount.delete(0, END)
    Entry_WeightsCount.delete(0,END)
    Entry_WeightsCount.insert(0,"{} gms".format(Total_weight))
    Entry_NodulesCount.insert(0,format(Total_nodules)) 
    Entry_Area.delete(0,END)
    Entry_Area.insert(0,"{} sq.meter".format(int(area)))
    Entry_CAmeraDist.delete(0,END)
    # Entry_CAmeraDist.insert(0,"{} mm".format(distance_seabed))
    # path=("D:/hindustan final project/New_App_Design/capture_image/")
    
    



def apply_change():
    global filename,img,Display_frames,f
    if f==0:
        f = inputtxt.get(1.0, "end-1c")
        print(f)
    else:
        f=750
    Flora_count=0
    if filename=="None":
        print('Please select File')
    else:
        Display_frames.destroy()
        
        Display_frames =tk.Label(Frame_Horizaontal_left_child_UL,bg="white",fg="black",borderwidth=0 ) # using label 
        Display_frames.pack(fill = BOTH , expand=1)
        print(filename)
        img=cv2.imread(filename)
        haar_cascade_browser()
        multiCas = cv2.resize(clrImg, (750, 500)) 
        image = Image.fromarray(multiCas)
        image=image.resize((750,500),Image.ANTIALIAS)
        imagetk = ImageTk.PhotoImage(image=image)
        Display_frames.image = imagetk
        Display_frames.configure(image=imagetk)
        path='D:/hindustan final project/New_App_Design/browser_image/'
        saved_image=cv2.imwrite(path+"frame-"+time.strftime("%d-%m-%Y-%H-%M-%S")+".jpg",cv2.cvtColor(clrImg,cv2.COLOR_RGB2BGR))
        new_path=path+"frame-"+time.strftime("%d-%m-%Y-%H-%M-%S")+".jpg"
        zoomed.Zoom_Advanced(Display_frames,new_path)
def open_file():
    global img, C1_count,Display_frames,filename
    Display_frames.destroy()
    
    Display_frames =tk.Label(Frame_Horizaontal_left_child_UL,bg="white",fg="black",borderwidth=0 ) # using label 
    Display_frames.pack(fill = BOTH , expand=1)
    
    
    f_types = [('Jpg Files', '*.jpg'),('all files', '*.*')]
    filename = filedialog.askopenfilename(filetypes=f_types)
    img=cv2.imread(filename)
    haar_cascade_browser()
    
    
    #p,l,m=cv2.split(img)
    #img=cv2.merge([m,l,p])
    multiCas = cv2.resize(clrImg, (750, 500)) 
    image = Image.fromarray(multiCas)
    image=image.resize((750,500),Image.ANTIALIAS)
    imagetk = ImageTk.PhotoImage(image=image)
    # Display_frames.image = imagetk
    # Display_frames.configure(image=imagetk)
    path='D:/hindustan final project/New_App_Design/browser_image/'
    saved_image=cv2.imwrite(path+"frame-"+time.strftime("%d-%m-%Y-%H-%M-%S")+".jpg",cv2.cvtColor(clrImg,cv2.COLOR_RGB2BGR))
    new_path=path+"frame-"+time.strftime("%d-%m-%Y-%H-%M-%S")+".jpg"
    zoomed.Zoom_Advanced(Display_frames,new_path)


def video_stream():
    global Display_frames,img
    haar_cascade()
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(10,10))
    
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    
    l, a, b = cv2.split(lab)  # split on 3 different channels
    # equ = cv2.equalizeHist(l)
    l2 = clahe.apply(l)  # apply CLAHE to the L-channel

    lab = cv2.merge((l2,a,b))  # merge channels
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)  # convert from LAB to RGB
    
    # cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    # ref_image_face_width = haar_cascade(img)
    # Focal_length_found = Focal_Length_Finder(
    #     Known_distance, Known_width, ref_image_face_width)   
    # font=cv2.FONT_HERSHEY_SIMPLEX
    # time.sleep(0.0001) 
    # time.sleep(1)
    # detection_buffer.put(frame)

    # Display_frames =tk.Label(Frame_Horizaontal_left_child_UL,bg="white",fg="black",borderwidth=0 ) # using label
    Display_frames.pack(fill = BOTH , expand=1)
    
    #p,l,m=cv2.split(img)
    #img=cv2.merge([p,l,m])
    
    multiCas = cv2.resize(img, (750,500))
    img = Image.fromarray(multiCas)
    img=img.resize((750,500)) # new width & height
    imgtk = ImageTk.PhotoImage(image=img)
    Display_frames.imgtk = imgtk
    Display_frames.configure(image=imgtk)
    Display_frames.after(6, video_stream)
    #cv2.imshow('img', multiCas)
# def threading_call():
    # if __name__=='__main__':
        
    #     t=threading.Thread(target= video_stream)
    #     t.start()
    # video_stream()
def open_img():
    global flag, C1_count,img
    haar_cascade()
    # ret, frame = cap.read()            
    path=("D:/hindustan final project/New_App_Design/capture_image/")            
    image=cv2.imread(path)
    # image = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)  
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(10,10))
    
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    
    l, a, b = cv2.split(lab)  # split on 3 different channels
    # equ = cv2.equalizeHist(l)
    l2 = clahe.apply(l)  # apply CLAHE to the L-channel

    lab = cv2.merge((l2,a,b))  # merge channels
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)  # convert from LAB to RGB
              
    #image = Image.fromarray(image)
    #image=image.resize((400,200),Image.ANTIALIAS)
    #imagetk = ImageTk.PhotoImage(image=image)
    
    
    #gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
       
    if ret:
        cv2.imwrite(path+"frame-"+time.strftime("%d-%m-%Y-%H-%M-%S")+".jpg",cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
    #p,l,m=cv2.split(image)
    #img=cv2.merge([m,l,p])
    multiCas = cv2.resize(image, (750,500)) 
    image = Image.fromarray(multiCas)
    image=image.resize((750,500),Image.ANTIALIAS)
    imagetk = ImageTk.PhotoImage(image=image)
    Display_frames.image = imagetk
    Display_frames.configure(image=imagetk)
    #cv2.imshow('img', multiCas)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    flag=3
    print(flag)

    return img

    
Button1_Capture = Button(Frame_Horizaontal_left_child_UR_child_top , text = "   Capture   ",bg = "red4" , fg = "white" ,relief="raised", command= open_img)
Button2_Video = Button(Frame_Horizaontal_left_child_UR_child_top , text = "     Video     ",bg = "red4" , fg = "white",relief="raised" , command= video_stream)
Button3_Browse = Button(Frame_Horizaontal_left_child_UR_child_top , text = "  Browse...   ",bg = "red4" , fg = "white",relief="raised", command= open_file )
Button4_Change = Button(Frame_Horizaontal_left_child_UR_child_top , text = " Apply change  ",bg = "red4" , fg = "white",relief="raised", command=apply_change)

Button1_Capture.grid(row = 0 , column = 0 , padx=25 , pady=10 , ipadx=10, ipady=5,sticky='nsew')
Button2_Video  .grid(row = 0 , column = 1 , padx=25 , pady=10 , ipadx=10, ipady=5,sticky='nsew')
Button3_Browse .grid(row = 0 , column = 2, padx=25, pady=10, ipadx=10, ipady=5,sticky='nsew')
Button4_Change .grid(row = 0 , column = 3, padx=25, pady=10, ipadx=10, ipady=5,sticky='nsew')

# t.join()

length_label1 = Label(Frame_Horizaontal_left_child_UR_child_bottom, text="Min Neighbor" , bg = "white")
length_label1.grid(row=0, column=0, pady=4, padx = 4)
SlideBar1 = Scale(Frame_Horizaontal_left_child_UR_child_bottom, from_=1, to=6, tickinterval= 1,
                  orient=HORIZONTAL, length=250, bg = "white",command=set_SliderBar1)
SlideBar1.set(3)
SlideBar1.grid(row=0, column=1)

length_label2 = Label(Frame_Horizaontal_left_child_UR_child_bottom, text="Scaling Factor", bg = "white")
length_label2.grid(row=1, column=0, pady=4, padx = 4)
SlideBar2 = Scale(Frame_Horizaontal_left_child_UR_child_bottom, from_=1.1, to=1.4, tickinterval= 0.05,
                  orient=HORIZONTAL,resolution=0.05, length=250, bg = "white",command=set_SliderBar2)
SlideBar2.set(1.2)
SlideBar2.grid(row=1, column=1)
length_label3 = Label(Frame_Horizaontal_left_child_UR_child_bottom, text="Focal Length", bg = "white")
length_label3.grid(row=2, column=0, pady=4, padx = 4)
inputtxt = Text(Frame_Horizaontal_left_child_UR_child_bottom,borderwidth=3,height = 1,width = 20)
inputtxt.grid(row=2, column=1)
'''
length_label4 = Label(Frame_Horizaontal_left_child_BR, text="Neighbour", bg = "white")./
    grid(row=3, column=0, pady=4, padx = 4)
SlideBar4 = Scale(Frame_Horizaontal_left_child_BR, from_=0, to=50, tickinterval= 50, /
                  orient=HORIZONTAL, length=250 , bg = "white")

SlideBar4.set(40)
SlideBar4.grid(row=3, column=1)
'''
# Label_NodulesCount = Label(Frame_Horizaontal_left_child_UR_child_bottom,text = "Distance" , bg = "white")
# Label_NodulesCount.grid(row=4, column=0, pady=4, padx = 4)
# Entry_NodulesCount = Entry(Frame_Horizaontal_left_child_UR_child_bottom,width = 30)
# Entry_NodulesCount.grid(row=4, column=1, pady=4, padx = 4)

# Label_NodulesCount = Label(Frame_Horizaontal_left_child_UR_child_bottom,text = "Width" , bg = "white")
# Label_NodulesCount.grid(row=5, column=0, pady=4, padx = 4)
# Entry_NodulesCount = Entry(Frame_Horizaontal_left_child_UR_child_bottom,width = 30)
# Entry_NodulesCount.grid(row=5, column=1, pady=4, padx = 4)

Label_NodulesCount = Label(Frame_Horizaontal_left_child_BL,text = "No of Fauna detected" , bg = "white")
Label_NodulesCount.grid(row=0, column=0, pady=4, padx = 4)
Entry_NodulesCount = Entry(Frame_Horizaontal_left_child_BL,width = 30)
Entry_NodulesCount.grid(row=0, column=1, pady=4, padx = 4)


Label_WeightsCount = Label(Frame_Horizaontal_left_child_BL,text = "Approx Size detected", bg = "white")
Label_WeightsCount.grid(row=1, column=0, pady=4, padx = 4)
Entry_WeightsCount = Entry(Frame_Horizaontal_left_child_BL,width = 30)
Entry_WeightsCount.grid(row=1, column=1, pady=4, padx = 4)



Label_CAmeraDist = Label(Frame_Horizaontal_left_child_BL,text = "Dist. b/w camera and seabed", bg = "white")
Label_CAmeraDist.grid(row=2, column=0, pady=4, padx = 4)
Entry_CAmeraDist = Entry(Frame_Horizaontal_left_child_BL,width = 30)
Entry_CAmeraDist.grid(row=2, column=1, pady=4, padx = 4)

Label_Area = Label(Frame_Horizaontal_left_child_BL,text = "Approx area covered", bg = "white")
Label_Area.grid(row=3, column=0, pady=4, padx = 4)
Entry_Area = Entry(Frame_Horizaontal_left_child_BL,width = 30)
Entry_Area.grid(row=3, column=1, pady=4, padx = 4)
Output = Text(Frame_Horizaontal_left_child_BL, height = 1,width = 45, bg = "white")
Output.grid(row=4, column=0, pady=4, padx = 4,columnspan=2)
# Output.insert(END,'Waiting for flora detection')
# Entry_NodulesCount.config(text = "85")

# Entry_WeightsCount.config(text = "76 kg")
# Entry_WeightsCount.insert(0,"76 kg")
# Entry_CAmeraDist.config(text = "28 ft")
# Entry_CAmeraDist.insert(0,"28 cm")

MainScreenGUI.mainloop()