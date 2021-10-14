from tkinter import *
from PIL import ImageTk, Image
from keras.models import *
from keras.preprocessing import image
import numpy as np
import cv2
from cv2 import *
import os
import keras.applications.imagenet_utils
from termcolor import colored
from tkinter import filedialog
from tkinter import messagebox

global lab
global mylabel
global mylabel1
global my_label3
global imgbk2
global exitimage
global Image_label
global browse_image

# window parameters
root = Tk()
root.title('Covid-19 Machine Learning Diagnosis')
root.geometry("500x668")
root.resizable(False, False)

p1 = ImageTk.PhotoImage(Image.open('Covid19.png'))

# Setting icon of master window
root.iconphoto(False, p1)

imgbk = ImageTk.PhotoImage(Image.open("resized.jpg"))
background_label = Label(root, image=imgbk)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

imgbk2 = ImageTk.PhotoImage(Image.open("resized2.jpg"))
exitimage = ImageTk.PhotoImage(Image.open("exit.png"))
browse_image = ImageTk.PhotoImage(Image.open("browse-folder.png"))

#Title for window
MyTitle = Label(root, text ="Covid Diagnosis program", fg ="black", font=("Arial", 20), bg="#76e5dc")
MyTitle.place(relx = 0.5, rely = 0.2, anchor = 'center')

# Label for single input
Single_label = Label(root, text="(Diagnosis for a single image)", font=("Arial", 10), bg="#76e5dc")
Single_label.place(relx = 0.53,rely = 0.4, anchor = 'center')

# Label for multiple input
multi_label = Label(root, text="(Diagnosis for a multiple images)", font=("Arial", 10), bg="#76e5dc")
multi_label.place(relx = 0.55,rely = 0.5, anchor = 'center')

# Label for directory input
multi2_label = Label(root, text="(Diagnosis for images in directory)", font=("Arial", 10), bg="#76e5dc")
multi2_label.place(relx = 0.55,rely = 0.6, anchor = 'center')

inputvalue = IntVar()
e = Entry(root, borderwidth= 5, width= 5, textvariable = inputvalue)
e.place(relx = 0.75, rely= 0.48)

def mywindow():
    window = Toplevel()
    window.title('Covid-19 Machine Learning Diagnosis')
    window.geometry("500x668")
    window.resizable(False, False)
    p1 = ImageTk.PhotoImage(Image.open('Covid19.png'))

    # Setting icon of master window
    window.iconphoto(False, p1)
    return window

def exit_btn():
    root.destroy()


def singlePage():
    global imgbk2
    root.iconify()
    window = mywindow()

    background_label = Label(window, image=imgbk2)
    background_label.place(x=0, y=0, relwidth=1, relheight=1)

    # Title for window
    MyTitle = Label(window, text="Single Input Diagnosis", fg="black", font=("Arial", 20), bg="#76e5dc")
    MyTitle.place(relx=0.5, rely=0.2, anchor='center')

    def exit_btn():
        window.destroy()

    def myClick():

        file_path = filedialog.askopenfilename()
        print(file_path)

        img1 = Image.open(file_path)
        img1.thumbnail((210, 210))
        img = cv2.imread(file_path)

        img1 = ImageTk.PhotoImage(img1)
        lab.configure(image=img1)
        lab.image = img1

        disease_class = ['Covid-19', 'Non Covid-19']
        disease_class0 = ['CT', 'Xray', 'Neither']
        img = cv2.resize(img, (224, 224))
        x = keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x /= 255

        modelctscan = load_model('CTDenseK3.h5')
        modelimage = load_model('Imodel.h5')
        modelxray2 = load_model('XmodelK4.h5')
        custom = modelimage.predict(x)
        a = custom[0]
        ind = np.argmax(a)
        print('Prediction:', disease_class0[ind])
        xray = 1
        ctscan = 0
        neither = 2

        if disease_class0[ind] == 'CT':
            image = ctscan
            print(colored('Image is ctscan', 'magenta', attrs=['bold']))
            z = "Image is CT"
            mylabel.config(text="Image is CT-scan!", font=("Arial", 12))
        elif disease_class0[ind] == 'Xray':
            image = xray
            print(colored('Image is xray', 'blue', attrs=['bold']))
            z = "Image is xray"
            mylabel.config(text="Image is Xray!", font=("Arial", 12))
        else:
            image = neither
            print(colored('Image is Neither CT or Xray', 'green', attrs=['bold']))
            z = "Image is Neither CT or Xray"
            mylabel.config(text="Image is Neither CT or Xray!", font=("Arial", 12))

        if image == 0:
            # print ("image b covid or normal if 1 = normal if 0 = covid////// output=",classes)
            custom = modelctscan.predict(x)
            a = custom[0]
            ind = np.argmax(a)
            print('Prediction:', disease_class[ind])
            if disease_class[ind] == 'Non Covid-19':
                print(colored('You are Non Covid', 'green', attrs=['bold']))
                mylabel1.config(text="You are Non Covid", font=("Arial", 12))
            else:
                print(colored('You are diagnosed with covid', 'red', attrs=['bold']))
                mylabel1.config(text="You are diagnosed with covid", font=("Arial", 12))

        elif image == 1:
            custom = modelxray2.predict(x)
            a = custom[0]
            ind = np.argmax(a)
            print('Prediction:', disease_class[ind])
            if disease_class[ind] == 'Non Covid-19':
                print(colored('You are Non Covid', 'green', attrs=['bold']))
                mylabel1.config(text="You are Non Covid", font=("Arial", 12))
            else:
                print(colored('You are diagnosed with covid', 'red', attrs=['bold']))
                mylabel1.config(text="You are diagnosed with covid", font=("Arial", 12))

        else:
            mylabel1.config(text="Please enter Either Chest Xray or Chest CT-Scan", font=("Arial", 12))


    # Labels for text and image
    mylabel = Label(window, text="Type of Image", fg="black", bg="#76e5dc", font=("Arial", 12))
    mylabel.place(relx=0.5, rely=0.25, anchor='center')
    mylabel1 = Label(window, text="Diagnosis result", fg="black", bg="#76e5dc", font=("Arial", 12))
    mylabel1.place(relx=0.5, rely=0.30, anchor='center')
    lab = Label(window, bg="#76e5dc")
    lab.place(relx=0.5, rely=0.49, anchor='center')

    start_button = Button(window, image=browse_image, command=myClick, borderwidth=5)
    start_button.place(relx=0.4, rely=0.72, anchor='center')

    exit_button = Button(window, image=exitimage, command=exit_btn, borderwidth=5)
    exit_button.place(relx=0.6, rely=0.72, anchor='center')
# End of Single Input.

def Multipage():
    global imgbk2
    global mylabelimagetype
    global mylabeldiagnosis
    global my_label3
    global image

    root.iconify()
    window2 = mywindow()

    background_label = Label(window2, image=imgbk2)
    background_label.place(x=0, y=0, relwidth=1, relheight=1)

    # Title for window
    MyTitle = Label(window2, text="Multi Input Diagnosis", fg="black", font=("Arial", 20), bg="#76e5dc")
    MyTitle.place(relx=0.5, rely=0.2, anchor='center')

    with open("inputuser.txt","w") as f:
        f.write(f"{e.get()}")

    with open('inputuser.txt') as f:
        lines = f.readlines()
    print(lines)

    input = int(lines[0])
    print(input)

    inputuser = input
    filess = []
    for i in range(0, inputuser):
        files = filedialog.askopenfilename()
        filess.append(files)
    stored = filess
    print(stored)


    for i in range(0, inputuser):
        if(stored[i]== ''):
            print("you must enter "+str(inputuser)+"images")
            messagebox.showerror(title="Warning!", message="you must enter "+str(inputuser)+" images")
            files = filedialog.askopenfilename()
            stored[i]= files

    print(stored)

    # taking the directory list >> opening image and storing in a list.
    image = []
    for i in range(0, inputuser):
        images = Image.open(stored[i])
        images = images.resize((210, 210))
        images = ImageTk.PhotoImage(images)
        image.append(images)
        # images[] is the output

    # to diagnose first image
    modelimage = load_model('Imodel.h5')
    modelctscan = load_model('CTDenseK3.h5')
    modelxray2 = load_model('XmodelK4.h5')
    disease_class = ['Covid-19', 'Non Covid-19']
    disease_class0 = ['CT-scan', 'X-ray', 'Neither']
    imageclass = []
    classtype = []
    for i in range(0, inputuser):
        img = cv2.imread(stored[i])
        img = cv2.resize(img, (224, 224))
        x = keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x /= 255
        custom = modelimage.predict(x)
        a = custom[0]
        ind = np.argmax(a)
        print('Prediction:', disease_class0[ind])
        xray = 1
        ctscan = 0
        neither = 2
        imageclass.append(disease_class0[ind])

        if disease_class0[ind] == 'CT-scan':
            imagev2 = ctscan
            print(colored('Image Type: Ct-scan', 'magenta', attrs=['bold']))
        elif disease_class0[ind] == 'X-ray':
            imagev2 = xray
            print(colored('Image Type: X-ray', 'blue', attrs=['bold']))
        else:
            imagev2 = neither
            print(colored('Image Type:  Neither CT or Xray', 'green', attrs=['bold']))

        if imagev2 == 0:
            # print ("image b covid or normal if 1 = normal if 0 = covid////// output=",classes)
            custom = modelctscan.predict(x)
            a = custom[0]
            ind = np.argmax(a)
            print('Prediction:', disease_class[ind])
            classtype.append(disease_class[ind])
            if disease_class[ind] == 'Non Covid-19':
                print(colored('Diagnosis: Non Covid!', 'green', attrs=['bold']))
            else:
                print(colored('Diagnosis: Covid-19!', 'red', attrs=['bold']))

        elif imagev2 == 1:
            custom = modelxray2.predict(x)
            a = custom[0]
            ind = np.argmax(a)
            print('Prediction:', disease_class[ind])
            classtype.append(disease_class[ind])
            if disease_class[ind] == 'Non Covid-19':
                print(colored('Diagnosis: Non Covid!', 'green', attrs=['bold']))
            else:
                print(colored('Diagnosis: Covid-19!', 'red', attrs=['bold']))

        else:
            classtype.append("Please enter Xray or CT-Scan")
            print("Please enter Either Chest Xray or Chest CT-Scan")

    print(imageclass)
    print(classtype)

    my_label3 = Label(window2, image=image[0], bg="#76e5dc")
    my_label3.place(relx=0.5, rely=0.5, anchor='center')

    # Labels for text and image
    mylabelimagetype = Label(window2, text="Image Type: " + imageclass[0] + "!", fg="black", bg="#76e5dc",
                             font=("Arial", 12))
    mylabelimagetype.place(relx=0.5, rely=0.25, anchor='center')
    mylabeldiagnosis = Label(window2, text="Diagnosis:" + classtype[0] + "!", fg="black", bg="#76e5dc", font=("Arial", 12))
    mylabeldiagnosis.place(relx=0.5, rely=0.30, anchor='center')

    def forward(image_number):
        global my_label3
        global button_forward
        global button_back
        global mylabelimagetype
        global mylabeldiagnosis

        my_label3.destroy()
        mylabelimagetype.destroy()
        mylabeldiagnosis.destroy()
        my_label3 = Label(window2, image=image[image_number - 1])
        mylabeldiagnosis = Label(window2, text="Diagnosis: " + classtype[image_number - 1] + "!", fg="black", bg="#76e5dc",
                                 font=("Arial", 12))
        mylabelimagetype = Label(window2, text="Image Type: " + imageclass[image_number - 1] + "!", fg="black",
                                 bg="#76e5dc", font=("Arial", 12))
        forward_button = Button(window2, text=">>", command=lambda: forward(image_number + 1))
        backward_button = Button(window2, text="<<", command=lambda: backward(image_number - 1))

        if image_number == input:
            forward_button = Button(window2, text=">>", state=DISABLED)

        my_label3.place(relx=0.5, rely=0.5, anchor='center')
        forward_button.place(relx=0.54, rely=0.70, anchor='center')
        backward_button.place(relx=0.45, rely=0.70, anchor='center')
        mylabelimagetype.place(relx=0.5, rely=0.25, anchor='center')
        mylabeldiagnosis.place(relx=0.5, rely=0.30, anchor='center')

    def backward(image_number):
        global my_label3
        global button_forward
        global button_back
        global mylabelimagetype
        global mylabeldiagnosis

        my_label3.destroy()
        mylabelimagetype.destroy()
        mylabeldiagnosis.destroy()
        my_label3 = Label(window2, image=image[image_number - 1])
        mylabeldiagnosis = Label(window2, text="Diagnosis: " + classtype[image_number - 1] + "!", fg="black", bg="#76e5dc",
                                 font=("Arial", 12))
        mylabelimagetype = Label(window2, text="Image Type: " + imageclass[image_number - 1] + "!", fg="black",
                                 bg="#76e5dc", font=("Arial", 12))
        forward_button = Button(window2, text=">>", command=lambda: forward(image_number + 1))
        backward_button = Button(window2, text="<<", command=lambda: backward(image_number - 1))

        if image_number == 1:
            backward_button = Button(window2, text="<<", state=DISABLED)

        my_label3.place(relx=0.5, rely=0.5, anchor='center')
        forward_button.place(relx=0.54, rely=0.70, anchor='center')
        backward_button.place(relx=0.45, rely=0.70, anchor='center')
        mylabelimagetype.place(relx=0.5, rely=0.25, anchor='center')
        mylabeldiagnosis.place(relx=0.5, rely=0.30, anchor='center')

    def exit_btn():
        window2.destroy()
        window2.update()

    # Button forward
    forward_button = Button(window2, text=">>", command=lambda: forward(2))
    forward_button.place(relx=0.54, rely=0.70, anchor='center')

    # Button backward
    backward_button = Button(window2, text="<<", command=backward, state=DISABLED)
    backward_button.place(relx=0.45, rely=0.70, anchor='center')

    exit_button = Button(window2, image=exitimage, command=exit_btn, borderwidth=5)
    exit_button.place(relx=0.5, rely=0.8, anchor='center')

def directory():
    global imgbk2
    global mylabelimagetype
    global mylabeldiagnosis
    global my_label
    global image

    root.iconify()
    window3 = mywindow()

    background_label = Label(window3, image=imgbk2)
    background_label.place(x=0, y=0, relwidth=1, relheight=1)

    # Title for window
    MyTitle = Label(window3, text="Directory Diagnosis", fg="black", font=("Arial", 20), bg="#76e5dc")
    MyTitle.place(relx=0.5, rely=0.2, anchor='center')

    file_path = filedialog.askdirectory()
    print(file_path)

    files = []
    cnt = 0
    for filename in os.listdir(file_path):
        img = os.path.join(file_path, filename)
        files.append(img)
        cnt = cnt + 1
    stored = files
    print(stored)
    print(cnt)

    image = []
    for i in range(0, cnt):
        images = Image.open(stored[i])
        images = images.resize((210, 210))
        images = ImageTk.PhotoImage(images)
        image.append(images)

    modelimage = load_model('Imodel.h5')
    modelctscan = load_model('CTDenseK3.h5')
    modelxray2 = load_model('XmodelK4.h5')
    disease_class = ['Covid-19', 'Non Covid-19']
    disease_class0 = ['CT-scan', 'X-ray', 'Neither']
    imageclass = []
    classtype = []
    for i in range(0, cnt):
        img = cv2.imread(stored[i])
        img = cv2.resize(img, (224, 224))
        x = keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x /= 255
        custom = modelimage.predict(x)
        a = custom[0]
        ind = np.argmax(a)
        print('Prediction:', disease_class0[ind])
        xray = 1
        ctscan = 0
        neither = 2
        imageclass.append(disease_class0[ind])

        if disease_class0[ind] == 'CT-scan':
            imagev2 = ctscan
            print(colored('Image Type: Ct-scan', 'magenta', attrs=['bold']))
        elif disease_class0[ind] == 'X-ray':
            imagev2 = xray
            print(colored('Image Type: X-ray', 'blue', attrs=['bold']))
        else:
            imagev2 = neither
            print(colored('Image Type:  Neither CT or Xray', 'green', attrs=['bold']))

        if imagev2 == 0:
            # print ("image b covid or normal if 1 = normal if 0 = covid////// output=",classes)
            custom = modelctscan.predict(x)
            a = custom[0]
            ind = np.argmax(a)
            print('Prediction:', disease_class[ind])
            classtype.append(disease_class[ind])
            if disease_class[ind] == 'Non Covid-19':
                print(colored('Diagnosis: Non Covid!', 'green', attrs=['bold']))
            else:
                print(colored('Diagnosis: Covid-19!', 'red', attrs=['bold']))

        elif imagev2 == 1:
            custom = modelxray2.predict(x)
            a = custom[0]
            ind = np.argmax(a)
            print('Prediction:', disease_class[ind])
            classtype.append(disease_class[ind])
            if disease_class[ind] == 'Non Covid-19':
                print(colored('Diagnosis: Non Covid!', 'green', attrs=['bold']))
            else:
                print(colored('Diagnosis: Covid-19!', 'red', attrs=['bold']))

        else:
            classtype.append("Please enter Xray or CT-Scan")
            print("Please enter Either Chest Xray or Chest CT-Scan")

    print(imageclass)
    print(classtype)

    my_label = Label(window3, image=image[0], bg="#76e5dc")
    my_label.place(relx=0.5, rely=0.5, anchor='center')

    # Labels for text and image
    mylabelimagetype = Label(window3, text="Image Type: " + imageclass[0] + "!", fg="black", bg="#76e5dc",
                             font=("Arial", 12))
    mylabelimagetype.place(relx=0.5, rely=0.25, anchor='center')
    mylabeldiagnosis = Label(window3, text="Diagnosis:" + classtype[0] + "!", fg="black", bg="#76e5dc", font=("Arial", 12))
    mylabeldiagnosis.place(relx=0.5, rely=0.30, anchor='center')

    def forward(image_number):
        global my_label
        global button_forward
        global button_back
        global mylabelimagetype
        global mylabeldiagnosis

        my_label.destroy()
        mylabelimagetype.destroy()
        mylabeldiagnosis.destroy()
        my_label = Label(window3, image=image[image_number - 1])
        mylabeldiagnosis = Label(window3, text="Diagnosis: " + classtype[image_number - 1] + "!", fg="black", bg="#76e5dc",
                                 font=("Arial", 12))
        mylabelimagetype = Label(window3, text="Image Type: " + imageclass[image_number - 1] + "!", fg="black",
                                 bg="#76e5dc", font=("Arial", 12))
        forward_button = Button(window3, text=">>", command=lambda: forward(image_number + 1))
        backward_button = Button(window3, text="<<", command=lambda: backward(image_number - 1))

        if image_number == cnt:
            forward_button = Button(window3, text=">>", state=DISABLED)

        my_label.place(relx=0.5, rely=0.5, anchor='center')
        forward_button.place(relx=0.54, rely=0.70, anchor='center')
        backward_button.place(relx=0.45, rely=0.70, anchor='center')
        mylabelimagetype.place(relx=0.5, rely=0.25, anchor='center')
        mylabeldiagnosis.place(relx=0.5, rely=0.30, anchor='center')

    def backward(image_number):
        global my_label
        global button_forward
        global button_back
        global mylabelimagetype
        global mylabeldiagnosis

        my_label.destroy()
        mylabelimagetype.destroy()
        mylabeldiagnosis.destroy()
        my_label = Label(window3, image=image[image_number - 1])
        mylabeldiagnosis = Label(window3, text="Diagnosis: " + classtype[image_number - 1] + "!", fg="black", bg="#76e5dc",
                                 font=("Arial", 12))
        mylabelimagetype = Label(window3, text="Image Type: " + imageclass[image_number - 1] + "!", fg="black",
                                 bg="#76e5dc", font=("Arial", 12))
        forward_button = Button(window3, text=">>", command=lambda: forward(image_number + 1))
        backward_button = Button(window3, text="<<", command=lambda: backward(image_number - 1))

        if image_number == 1:
            backward_button = Button(window3, text="<<", state=DISABLED)

        my_label.place(relx=0.5, rely=0.5, anchor='center')
        forward_button.place(relx=0.54, rely=0.70, anchor='center')
        backward_button.place(relx=0.45, rely=0.70, anchor='center')
        mylabelimagetype.place(relx=0.5, rely=0.25, anchor='center')
        mylabeldiagnosis.place(relx=0.5, rely=0.30, anchor='center')

    def exit_btn():
        window3.destroy()
        window3.update()

    # Button forward
    forward_button = Button(window3, text=">>", command=lambda: forward(2))
    forward_button.place(relx=0.54, rely=0.70, anchor='center')

    # Button backward
    backward_button = Button(window3, text="<<", command=backward, state=DISABLED)
    backward_button.place(relx=0.45, rely=0.70, anchor='center')

    exit_button = Button(window3, image=exitimage, command=exit_btn, borderwidth=5)
    exit_button.place(relx=0.5, rely=0.8, anchor='center')




# button for moving to single mode
single_button = Button(root, command=singlePage, borderwidth = 5,text="Single image")
single_button.place(relx = 0.25, rely = 0.4, anchor = 'center')

# Button for moving to multi mode
multi_button = Button(root, command=Multipage, borderwidth = 5,text="Multi image")
multi_button.place(relx = 0.25, rely = 0.5, anchor = 'center')

multi2_button = Button(root, command=directory, borderwidth = 5,text="   Directory   ")
multi2_button.place(relx = 0.25, rely = 0.6, anchor = 'center')

exit_button = Button(root, image=exitimage, command=exit_btn, borderwidth=5)
exit_button.place(relx=0.5, rely=0.72, anchor='center')

mainloop()