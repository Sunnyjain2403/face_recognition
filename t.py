from tkinter import *
import cv2
import csv
import numpy as np

import os
from PIL import Image
master = Tk() 
Label(master, text='roll no').place(x=200,y=100) 
Label(master, text='name').place(x=200,y=140)
Label(master, text='branch').place(x=200,y=180)
Label(master, text='roll no').place(x=200,y=400)
master.geometry('600x600')
roll=StringVar()
name=StringVar()
branch=StringVar()
idtodelete=StringVar()
e1 = Entry(master,textvar=roll) 
e2 = Entry(master,textvar=name)
e3 = Entry(master,textvar=branch)
e4 = Entry(master,textvar=idtodelete)
e1.place(x=300,y=100)
e2.place(x=300,y=140)
e3.place(x=300,y=180)
e4.place(x=300,y=400)


def inserts():
    i=roll.get()
    n=name.get()
    b=branch.get()
    if(i.isnumeric() and n.isalpha() and b.isalpha()):
        cam = cv2.VideoCapture(0)
        detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        sample=0
        while(True):
            _,frame = cam.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.1, 4)
            for (x,y,w,h) in faces:
                
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                sample=sample+1                
                cv2.imwrite("trainingimages\ "+n.lower()+"."+b.lower()+"."+i+"."+str(sample)+".jpg", gray[y:y+h,x:x+w])
            cv2.imshow('frame',frame)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            elif sample>60:
                break
        cam.release()
        cv2.destroyAllWindows()
        with open('db.csv','a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow([i,n,b])
        csvFile.close()
def faceid(path):
    images=[os.path.join(path,x) for x in os.listdir(path)]
    faces=[]
    ids=[]
    for image in images:
        curimage=Image.open(image).convert('L')
        face=np.array(curimage,'uint8')
        print(os.path.split(image))
        id=int(os.path.split(image)[-1].split(".")[2])
        ids.append(id)
        faces.append(face)

    return faces,ids


def train():
    recog=cv2.face.LBPHFaceRecognizer_create()
    faces,ids=faceid("trainingimages")
    recog.train(faces,np.array(ids))
    recog.save("trained.yml")
def idenitify():
    


    recog=cv2.face.LBPHFaceRecognizer_create()
    recog.read("trained.yml")
    cam=cv2.VideoCapture(0)
    file=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    while(True):
        _,frame=cam.read()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=file.detectMultiScale(gray,1.1,4)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            Id, conf = recog.predict(gray[y:y+h,x:x+w])
            
            if conf<90:
                s="unknown"
                with open('db.csv', newline='') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        if row['id']==str(Id):
                            s=row['id']+row['name']+row['branch']   
                cv2.putText(frame,s,(x+w,y+h),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),2)
        cv2.imshow('frame2',frame)
        if cv2.waitKey(100) & 0xFF==ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()
    with open('db.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            print(row)
    
def exited():
    exit()
def deleteid():
    z=idtodelete.get()
    if (z.isnumeric()):
        lines = list()
        with open('db.csv', 'r') as readFile:
            reader = csv.reader(readFile)
            for row in reader:
                if row[0]!=str(z):
                    lines.append(row)
        with open('db.csv', 'w', newline='') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerows(lines)

button1=Button(master,text="Submit",width=20,fg='#fff',bg='red',font=("roboto",15,"bold"),command=inserts)
button1.place(x=175,y=220)
button2=Button(master,text="train",width=20,fg='#fff',bg='red',font=("roboto",15,"bold"),command=train)
button2.place(x=175,y=270)
button3=Button(master,text="idenitify",width=20,fg='#fff',bg='red',font=("roboto",15,"bold"),command=idenitify)
button3.place(x=175,y=320)
button4=Button(master,text="delete",width=20,fg='#fff',bg='red',font=("roboto",15,"bold"),command=deleteid)
button4.place(x=175,y=450)
button5=Button(master,text="exit",width=20,fg='#fff',bg='red',font=("roboto",15,"bold"),command=exited)
button5.place(x=175,y=500)



mainloop()




