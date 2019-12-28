import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import img_math,cv2,os

class App(ttk.Frame):
    width = 750   #宽
    heigh = 400   #高
    def __init__(self, win):
        ttk.Frame.__init__(self,win)
        self.pack()
        win.title("车牌识别系统")
        win.geometry('+300+200')
        win.minsize(App.width,App.heigh)

        frame_1 = ttk.Frame(self)
        frame_1.grid(column=0, row=0)

        frame_2 = ttk.Frame(self)
        frame_2.grid(column=1, row=0)

        frame_3 = ttk.Frame(self)
        frame_3.grid(column=2, row=0)

        frame_4 = ttk.Frame(self)
        frame_4.grid(column=0, row=1)

        

        #显示分离后的车牌字符
        frame_5111 = ttk.Frame(self)
        frame_5111.grid(column=1, row=1)

        #显示分离后的车牌字符
        frame_5 = ttk.Frame(frame_5111)
        frame_5.pack()
        frame_5222 = ttk.Frame(frame_5111)
        frame_5222.pack()


        frame_6 = ttk.Frame(self)
        frame_6.grid(column=2, row=1)

        self.image_1 = ttk.Label(frame_1)
        self.image_1.pack()
        self.image_11 = ttk.Label(frame_1,text='灰度变化',font=('Times', '14'))
        self.image_11.pack()

        self.image_2 = ttk.Label(frame_2)
        self.image_2.pack()
        self.image_22 = ttk.Label(frame_2,text='边缘检测',font=('Times', '14'))
        self.image_22.pack()

        self.image_3 = ttk.Label(frame_3)
        self.image_3.pack()
        self.image_33 = ttk.Label(frame_3,text='形态学处理',font=('Times', '14'))
        self.image_33.pack()

        self.image_4 = ttk.Label(frame_4)
        self.image_4.pack()
        self.image_44 = ttk.Label(frame_4,text='车牌定位',font=('Times', '14'))
        self.image_44.pack()

        self.image_5_1 = ttk.Label(frame_5)
        self.image_5_1.grid(column=0, row=0)
        self.image_5_2 = ttk.Label(frame_5)
        self.image_5_2.grid(column=1, row=0)
        self.image_5_3 = ttk.Label(frame_5)
        self.image_5_3.grid(column=2, row=0)
        self.image_5_4 = ttk.Label(frame_5)
        self.image_5_4.grid(column=3, row=0)
        self.image_5_5 = ttk.Label(frame_5)
        self.image_5_5.grid(column=4, row=0)
        self.image_5_6 = ttk.Label(frame_5)
        self.image_5_6.grid(column=5, row=0)
        self.image_5_7 = ttk.Label(frame_5)
        self.image_5_7.grid(column=6, row=0)
        self.image_5_8 = ttk.Label(frame_5222,text='字符分割',font=('Times', '14'))
        self.image_5_8.pack()

        # self.image_6 = ttk.Label(frame_6)
        # self.image_6.pack()
        chu = ttk.Button(
            frame_6, text="退出", width=20, command=self.close_window)
        chu.grid(column=0, row=2)
        self.jiazai()

    def get_imgtk(self, img_bgr):
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(img)
        pil_image_resized = im.resize((250,170),Image.ANTIALIAS)
        imgtk = ImageTk.PhotoImage(image=pil_image_resized)
        return imgtk

    def get_imgtk_1(self, img_bgr):
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(img)
        pil_image_resized = im.resize((30,30),Image.ANTIALIAS)
        imgtk = ImageTk.PhotoImage(image=pil_image_resized)
        return imgtk

    def jiazai(self):
        img_1 = img_math.img_read("tmp/img_gray.jpg")
        self.img1 = self.get_imgtk(img_1)
        self.image_1.configure(image=self.img1)

        img_2 = img_math.img_read("tmp/img_edge.jpg")
        self.img2 = self.get_imgtk(img_2)
        self.image_2.configure(image=self.img2)

        img_3 = img_math.img_read("tmp/img_xingtai.jpg")
        self.img3 = self.get_imgtk(img_3)
        self.image_3.configure(image=self.img3)

        img_4 = img_math.img_read("tmp/img_caijian.jpg")
        self.img4 = self.get_imgtk(img_4)
        self.image_4.configure(image=self.img4)


        img_5_1 = img_math.img_read("tmp/chechar1.jpg")
        self.img51 = self.get_imgtk_1(img_5_1)
        self.image_5_1.configure(image=self.img51)

        img_5_2 = img_math.img_read("tmp/chechar2.jpg")
        self.img52 = self.get_imgtk_1(img_5_2)
        self.image_5_2.configure(image=self.img52)

        img_5_3 = img_math.img_read("tmp/chechar3.jpg")
        self.img53 = self.get_imgtk_1(img_5_3)
        self.image_5_3.configure(image=self.img53)

        img_5_4 = img_math.img_read("tmp/chechar4.jpg")
        self.img54 = self.get_imgtk_1(img_5_4)
        self.image_5_4.configure(image=self.img54)

        img_5_5 = img_math.img_read("tmp/chechar5.jpg")
        self.img55 = self.get_imgtk_1(img_5_5)
        self.image_5_5.configure(image=self.img55)

        img_5_6 = img_math.img_read("tmp/chechar6.jpg")
        self.img56 = self.get_imgtk_1(img_5_6)
        self.image_5_6.configure(image=self.img56)

        img_5_7 = img_math.img_read("tmp/chechar7.jpg")
        self.img57 = self.get_imgtk_1(img_5_7)
        self.image_5_7.configure(image=self.img57)

    def close_window(self):
        uu = ['tmp/'+i for i in os.listdir('tmp/')]
        for i in uu:
            os.remove(i)
        print("destroy")
        root.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()