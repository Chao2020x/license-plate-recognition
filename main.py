import cv2,os
from tkinter.filedialog import askopenfilename
from tkinter import ttk
import tkinter as tk
from PIL import Image, ImageTk
import img_function as predict
import img_math as img_math
import img_recognition as img_rec


class UI_main(ttk.Frame):
    pic_path = ""   #图片路径
    pic_source = "" 
    colorimg = 'white'   #车牌颜色
    cameraflag = 0
    width = 700   #宽
    height = 400   #高
    color_transform = img_rec.color_tr

    def __init__(self, win):
        ttk.Frame.__init__(self, win)

        
        win.title("车牌识别系统")
        win.geometry('+300+200')
        win.minsize(UI_main.width,UI_main.height)
        win.configure(relief=tk.RIDGE)
        # win.update()
        

        self.pack(fill=tk.BOTH)
        frame_left = ttk.Frame(self)
        frame_right_1 = ttk.Frame(self)
        frame_right_2 = ttk.Frame(self)
        frame_left.pack(side=tk.LEFT, expand=1, fill=tk.BOTH)
        frame_right_1.pack(side=tk.TOP, expand=1, fill=tk.Y)
        frame_right_2.pack()

        #界面左边 --->车牌识别主界面大图片
        self.image_ctl = ttk.Label(frame_left)
        self.image_ctl.pack()

        #界面右边 --->定位车牌位置、识别结果
        ttk.Label(frame_right_1, text='定位车牌：', font=('Times', '14')).grid(
            column=0, row=6, sticky=tk.NW)

        
        self.roi_ct2 = ttk.Label(frame_right_1)
        self.roi_ct2.grid(column=0, row=7, sticky=tk.W,pady=5)
        ttk.Label(frame_right_1, text='识别结果：', font=('Times', '14')).grid(
            column=0, row=8, sticky=tk.W,pady=5)
        self.r_ct2 = ttk.Label(frame_right_1, text="", font=('Times', '20'))
        self.r_ct2.grid(column=0, row=9, sticky=tk.W,pady=5)
        
        #车牌颜色
        self.color_ct2 = ttk.Label(frame_right_1,background=self.colorimg, 
                        text="", width="4",font=('Times', '14'))
        self.color_ct2.grid(column=0, row=10, sticky=tk.W)
        
        #界面右下角
        from_pic_ctl = ttk.Button(
            frame_right_2, text="车牌图片", width=20, command=self.from_pic)
        from_pic_ctl.grid(column=0, row=1)
        
        #清除识别数据
        from_pic_chu = ttk.Button(
            frame_right_2, text="清除识别数据", width=20, command=self.clean)
        from_pic_chu.grid(column=0, row=2)
        #查看图像处理过程
        from_pic_chu = ttk.Button(
            frame_right_2, text="查看图像处理过程", width=20, command=self.pic_chuli)
        from_pic_chu.grid(column=0, row=3)

        

        self.clean()

        self.predictor = predict.CardPredictor()
        self.predictor.train_svm()

    def get_imgtk(self, img_bgr):
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(img)
        pil_image_resized = im.resize((500,400),Image.ANTIALIAS)
        imgtk = ImageTk.PhotoImage(image=pil_image_resized)
        return imgtk

    #显示图片处理过程
    def pic_chuli(self):
        os.system("python ./chuli.py")

    def pic(self, pic_path):
        img_bgr = img_math.img_read(pic_path)
        first_img, oldimg = self.predictor.img_first_pre(img_bgr)
        if not self.cameraflag:
            self.imgtk = self.get_imgtk(img_bgr)
            self.image_ctl.configure(image=self.imgtk)
        r_color, roi_color, color_color = self.predictor.img_only_color(oldimg, 
                                                                oldimg, first_img)
        self.color_ct2.configure(background=color_color)
        try:
            Plate = HyperLPR_PlateRecogntion(img_bgr)
            r_color = Plate[0][0]
        except:
            pass
        self.show_roi(r_color, roi_color, color_color)
        self.colorimg = color_color
        print("|", color_color,
              r_color, "|", self.pic_source)

    #来自图片--->打开系统接口获取图片绝对路径
    def from_pic(self):
        self.cameraflag = 0
        self.pic_path = askopenfilename(title="选择识别图片", filetypes=[(
            "图片", "*.jpg;*.jpeg;*.png")])
        
        self.clean()
        self.pic_source = "本地文件：" + self.pic_path
        self.pic(self.pic_path)
        print(self.colorimg)
           
    def show_roi(self, r, roi, color):  # 车牌定位后的图片
        if r:
            try:
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                roi = Image.fromarray(roi)
                pil_image_resized = roi.resize((200, 50), Image.ANTIALIAS)
                self.tkImage2 = ImageTk.PhotoImage(image=pil_image_resized)
                self.roi_ct2.configure(image=self.tkImage2, state='enable')
            except:
                pass
            self.r_ct2.configure(text=str(r))
            try:
                c = self.color_transform[color]
                self.color_ct2.configure(text=c[0], state='enable')
            except:
                self.color_ct2.configure(state='disabled')
   
    #清除识别数据,还原初始结果
    def clean(self):
        img_bgr3 = img_math.img_read("pic/hy.png")
        self.imgtk2 = self.get_imgtk(img_bgr3)
        self.image_ctl.configure(image=self.imgtk2)

        self.r_ct2.configure(text="")
        self.color_ct2.configure(text="", state='enable')
        #显示车牌颜色
        self.color_ct2.configure(background='white' ,text="颜色", state='enable')
        self.pilImage3 = Image.open("pic/locate.png")
        pil_image_resized = self.pilImage3.resize((200, 50), Image.ANTIALIAS)
        self.tkImage3 = ImageTk.PhotoImage(image=pil_image_resized)
        self.roi_ct2.configure(image=self.tkImage3, state='enable')


if __name__ == '__main__':
    win = tk.Tk()

    ui_main= UI_main(win)
    # 进入消息循环
    win.mainloop()
