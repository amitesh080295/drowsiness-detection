import tkinter as tk
import customtkinter as ctk
import torch
import numpy as np
import cv2
import vlc
import random
from PIL import Image, ImageTk

app = tk.Tk()
app.geometry('600x600')
app.title('Alert Eyes')
ctk.set_appearance_mode('dark')

vid_frame = tk.Frame(height=480, width=600)
vid_frame.pack()
vid = ctk.CTkLabel(vid_frame)
vid.pack()

counter = 0
counter_label = ctk.CTkLabel(text=counter, height=40, width=120, text_color='white', fg_color='teal')
counter_label.pack(pady=10)

def reset_counter():
    global counter
    counter = 0

reset_button = ctk.CTkButton(text='Reset Counter', command=reset_counter, height=40, width=120, text_color='white', fg_color='teal')
reset_button.pack()

model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp9/weights/last.pt', force_reload=True)
cap = cv2.VideoCapture(0)

def detect():
    global counter
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame)
    img = np.squeeze(results.render())

    if len(results.xywh[0] > 0):
        detected_conf = results.xywh[0][0][4]
        detected_class = results.xywh[0][0][5]

        if detected_conf.item() > 0.85 and detected_class.item() == 1.0:
            file_choice = random.choice([1, 2, 3])
            p = vlc.MediaPlayer(f'file:///{file_choice}.wav')
            p.play()
            counter += 1

    img_arr = Image.fromarray(img)
    img_tk = ImageTk.PhotoImage(img_arr)
    vid.imgtk = img_tk
    vid.configure(image=img_tk)
    vid.after(10, detect)
    counter_label.configure(text=counter)

detect()
app.mainloop()