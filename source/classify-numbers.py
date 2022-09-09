# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 17:15:16 2021

@author: Sivasangaran V
"""
import os
import cv2
import numpy as np
import argparse
ROI_top = 100
ROI_bottom = 300
ROI_right = 150
ROI_left = 350


def parse_args():
    parser=argparse.ArgumentParser("Welcome to Sign Language Recognition")
    parser.add_argument("-model_path",help="Model Path")
    parser.add_argument("-temp",help="Temporary Directory Path")
    parser.add_argument("-mail",help="Whether you wish to mail the output",default="NO")
    return parser.parse_args()


import smtplib, ssl


def sendMessage(message,sender_email,receiver_email,password,smtp_server="smtp.gmail.com",port=587):
    context = ssl.create_default_context()
    try:
        server = smtplib.SMTP(smtp_server,port)
        server.ehlo() # Can be omitted
        server.starttls(context=context) # Secure the connection
        server.ehlo() # Can be omitted
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)
    except Exception as e:
            # Print any error messages to stdout
        print(e)
    finally:
        server.quit() 



num = {0:"0",1:"1",2:"2",3:"3",4:"4",5:"5",6:"6",7:"7",8:"8",9:"9"}
i=0
args=parse_args()
model_path=args.model_path;
temp=args.temp;

cap = cv2.VideoCapture(0)

panel = np.zeros([100, 700], np.uint8)
cv2.namedWindow('panel')

def nothing(x):
    pass

cv2.createTrackbar('L - h', 'panel', 0, 179, nothing)
cv2.createTrackbar('U - h', 'panel', 179, 179, nothing)

cv2.createTrackbar('L - s', 'panel', 0, 255, nothing)
cv2.createTrackbar('U - s', 'panel', 255, 255, nothing)

cv2.createTrackbar('L - v', 'panel', 0, 255, nothing)
cv2.createTrackbar('U - v', 'panel', 255, 255, nothing)

cv2.createTrackbar('S ROWS', 'panel', 0, 480, nothing)
cv2.createTrackbar('E ROWS', 'panel', 480, 480, nothing)
cv2.createTrackbar('S COL', 'panel', 0, 640, nothing)
cv2.createTrackbar('E COL', 'panel', 640, 640, nothing)
num_frames=0
while True:
    _, frame = cap.read()
    frame = cv2.flip(frame,1)
    s_r = cv2.getTrackbarPos('S ROWS', 'panel')
    e_r = cv2.getTrackbarPos('E ROWS', 'panel')
    s_c = cv2.getTrackbarPos('S COL', 'panel')
    e_c = cv2.getTrackbarPos('E COL', 'panel')

    roi = frame[s_r: e_r, s_c: e_c]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos('L - h', 'panel')
    u_h = cv2.getTrackbarPos('U - h', 'panel')
    l_s = cv2.getTrackbarPos('L - s', 'panel')
    u_s = cv2.getTrackbarPos('U - s', 'panel')
    l_v = cv2.getTrackbarPos('L - v', 'panel')
    u_v = cv2.getTrackbarPos('U - v', 'panel')

    lower_green = np.array([l_h, l_s, l_v])
    upper_green = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask_inv = cv2.bitwise_not(mask)

    bg = cv2.bitwise_and(roi, roi, mask=mask)
    fg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    cv2.imshow('bg', bg)
    cv2.imshow('fg', fg)

    cv2.imshow('panel', panel)
    num_frames+=1
    k = cv2.waitKey(30) & 0xFF
    #Commented out on 25/2/2021, 13:50
    '''if k == 27:
        cv2.imwrite("C:\\Users\\Sivasangaran V\\Desktop\\FYP\\FYP\\temp.jpg",fg)
        break'''
    #Created on 25/2/2021, 13:53
    if k == 27:
        i=i+1
        cv2.imwrite(os.path.join(temp,"{}.jpg".format(i)),fg)
        break
    if k == ord('n') or k == ord('N'):
        i=i+1
        cv2.imwrite(os.path.join(temp,"{}.jpg".format(i)),fg)
        continue
cap.release()
cv2.destroyAllWindows()
#Number of images captured = i
#Starting image = 1
#Iteration from j = 1 to i
#Which means j = range(1,i+1)
import tensorflow as tf
from tensorflow import keras

model_dl = keras.models.load_model(model_path) #look for local saved file
batch_size = 32
img_height = 64
img_width = 64
from keras.preprocessing import image

#Creating a dictionary to map each of the indexes to the corresponding number or letter

dict = num
res=''
#Predicting images
for j in range(1,i+1):
#img = image.load_img(ds_asl_dir+"\\a\\hand1_a_bot_seg_1_cropped.jpeg", target_size=(img_width, img_height))
    img = image.load_img(os.path.join(temp,"{}.jpg".format(j)), target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    _image = np.vstack([x])
    classes = model_dl.predict_classes(_image, batch_size=batch_size)
    probabilities = model_dl.predict_proba(_image, batch_size=batch_size)
    probabilities_formatted = list(map("{:.2f}%".format, probabilities[0]*100))

    print(classes) #displaying matrix prediction position
    res=res+dict[classes.item()]
    #print(f'The predicted image corresponds to "{dict[classes.item()]}" with {probabilities_formatted[classes.item()]} probability.') #displaying matrix prediction position name (number or letter) #commented out on 25/2/2021, 14:59

    os.remove(os.path.join(temp,"{}.jpg".format(j)))
flag=0
print("+----------------------------------------------+")
print("Message is {}".format(res))
print("+----------------------------------------------+")
print("Thank You")
message="""Subject: Message: {}\n\nDear User,\nThank you for using our product""".format(res)
if args.mail.upper() == "YES":
    smtp_server="smtp.gmail.com"
    port=587
    print("Enter your mail id: ")
    sender_email=input()
    print("Enter password: ")
    password=input()
    print("Whom do you want to send? ")
    receiver_email=input()       
    context = ssl.create_default_context()
    try:
        server = smtplib.SMTP(smtp_server,port)
        server.ehlo() # Can be omitted
        server.starttls(context=context) # Secure the connection
        server.ehlo() # Can be omitted
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)
    except Exception as e:
            # Print any error messages to stdout
        print(e)
        flag=1
        os.system("pause")
    finally:
        server.quit() 
        if flag == 0:
            print("Mail sent successfully to {}".format(receiver_email))
            os.system("pause")
