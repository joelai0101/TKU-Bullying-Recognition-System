#!/usr/bin/env python
# coding: utf-8

# # main

# In[1]:



import ICNP as ICNP
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
#from keras import activations
from keras import layers

model2 = Sequential()
model2.add(LSTM(72, input_shape=(6, 72)))
#model.add(Dropout(0.5))
model2.add(layers.Dense(512, activation='relu'))
model2.add(Dropout(0.3))
model2.add(layers.Dense(50, activation='relu'))
model2.add(Dropout(0.3))
model2.add(layers.Dense(1, activation='sigmoid'))
model2.summary()

model2.load_weights("model325.h5")


# In[2]:


def run():
    import ICNP as ICNP
    #main
    #相片List
    img_list=[]
    img_list=ICNP.img_capture()
    #文字list
    x_test=[]
    for i in img_list:
        x_test.append(ICNP.node_print(i))
        #print(x_test)
    x_test=ICNP.np.array(x_test)
    print(x_test.shape)

    try1=x_test.reshape(10,6,72)#10秒,每秒 6個幀,每張有72個節點
    print(try1.shape)
    pred=[]
    pred_new=[]

    for i in range(10):
        x_test_1=try1[i].reshape(1,6,72)
        pred.append(model2.predict(x_test_1))
    print(pred)

    for b in pred:
        pred_new.append(int(b*100))

    print(pred_new)
    
    global s
    global img
    img=img_list
    s=pred_new


# In[24]:


def GUI():
    
    import cv2
    a=[]
    i=0
    pos=0
    now=0
    f=0
    text1="no bully"
    pred_new=s
    img_list=img
    
    def nil(x):
        pass

    cv2.namedWindow('Trackbar')#windowname
    cv2.resizeWindow('Trackbar',800,600)
    cv2.createTrackbar('time','Trackbar',0,59,nil)#barname,windowname,startvalue,stopvalue,returntofunction
    cv2.createTrackbar('predict','Trackbar',0,100,nil)
    while(True):

        cv2.setTrackbarPos('predict', 'Trackbar', pred_new[int(now/6)])

        pos=cv2.getTrackbarPos('time','Trackbar')

        if now==pos:
            now=now+1
            pos=pos+1
            cv2.setTrackbarPos('time', 'Trackbar', now)#barname,windowname,setbarposition 

        else:

            now=pos
            #cap=img_list#不能動到原img_list
            cap=cv2.resize(img_list[now],(0,0),fx=1.5,fy=1.5)
            cv2.imshow('Trackbar',cap)
            
        
        if pred_new[int(now/6)-1] >= 50:
            f=f+1
            if f==3:
                f=2
        else:
            f=f-1
            if f==-1:
                f=0
        if f==2:
            text1="bully"
        if f==0:
            text1="no bully"
        #cap=img_list
        cap=cv2.resize(img_list[now-1],(0,0),fx=1.5,fy=1.5)
        if now==60:
            now=59
        cv2.putText(cap,text1,(600,60),cv2.FONT_HERSHEY_PLAIN,5,(0,0,206),5,cv2.LINE_AA)
        cv2.imshow('Trackbar',cap)






        if  cv2.waitKey(166)==ord('q') or now==60 : #
            break#fps,影片結束
    #cv2.waitKey(0)
    cv2.destroyAllWindows()


# In[26]:


import tkinter as tk
import cv2
root=tk.Tk()
tk.Button(root,text='run',activebackground='red',height=10,command=run).pack()
tk.Button(root,text='GUI',activebackground='red',height=10,command=GUI).pack()
root.mainloop()


# In[48]:


#只測試相機時

img_list=[]
img_list=ICNP.img_capture()
img=img_list
s=[10,10,10,10,10,10,10,10,10,10]


# In[ ]:




