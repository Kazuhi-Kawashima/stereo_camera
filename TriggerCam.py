import tkinter as tk
from tkinter import ttk
from datetime import datetime
import struct

import imagingcontrol4 as ic4
import cv2
from PIL import Image, ImageTk
from pymodbus.client import ModbusTcpClient as ModbusClient

#test
#test2
class CameraApp(tk.Tk):
    def __init__(self):

        ic4.Library.init()
        self.left_grabber = ic4.Grabber()
        self.right_grabber = ic4.Grabber()
        self.left_sink = ic4.SnapSink()
        self.right_sink = ic4.SnapSink()

        tk.Tk.__init__(self)
        self.title("Camera App")
        self.geometry("1920x1080")
        self.grid_rowconfigure(index=0, weight=1)
        self.grid_rowconfigure(index=1, weight=1)
        self.grid_rowconfigure(index=2, weight=1)
        self.grid_rowconfigure(index=3, weight=1)
        self.grid_rowconfigure(index=4, weight=1)
        self.grid_rowconfigure(index=5, weight=1)
        self.grid_rowconfigure(index=6, weight=1)
        self.grid_rowconfigure(index=7, weight=1)
        self.grid_rowconfigure(index=8, weight=1)
        self.grid_rowconfigure(index=9, weight=1)
        self.grid_rowconfigure(index=10, weight=1)
        self.grid_columnconfigure(index=0, weight=2)
        self.grid_columnconfigure(index=1, weight=4)
        self.grid_columnconfigure(index=2, weight=4)

        self.capture_flag_address = 15 #画像キャプチャフラグのアドレス
        self.complete_flag_address = 48 #完了フラグのアドレス
        self.register_address = 132 #物体座標送信用のレジスタのアドレス始点
        self.dummy_data=[12.1,12.1,12.1,12.1,12.1,12.1] #送信用ダミーデータ
        self.update_interval = 20 # 画像更新間隔
        self.check_interval = 1000 # DO8のチェック間隔
        self.modbus_server_ip ='127.0.0.1' #ModbusサーバーのIPアドレス
        self.modbus_server_port =502 #Modbusサーバーのポート番号
        self.flag_is_checked =False
        self.devices = []
        self.client = None  
        self.create_widgets()
        self.load_devices()      

    def create_widgets(self):
        "ウィジェットの作成"
        # カメラ選択用のコンボボックス
        tk.Label(self, text="left camera").grid(row=0, column=1, padx=10, pady=5, sticky='w')
        self.comboBoxDevices1 = ttk.Combobox(self, width=20)
        self.comboBoxDevices1.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        tk.Label(self, text="right camera").grid(row=0, column=2, padx=10, pady=5, sticky='w')
        self.comboBoxDevices2 = ttk.Combobox(self, width=20)
        self.comboBoxDevices2.grid(row=1, column=2, padx=5, pady=5, sticky="ew")

        # Mカメラ接続用の接続ボタン
        self.camera_connect_btn = tk.Button(self, text="camera connect", command=self.connect_cameras)
        self.camera_connect_btn.grid(row=2, column=0, padx=5, pady=5, sticky="ew")

        # Modbusサーバー接続用の接続ボタン
        self.buttonConnect = tk.Button(self, text="Connect", command=self.connect_modbus_server)
        self.buttonConnect.grid(row=3, column=0, padx=5, pady=5, sticky="ew")

        # カメラ映像を表示するcanvas
        self.label1 = tk.Label(self)
        self.label1.grid(row=3, column=1, rowspan=4, sticky="nsew")
        self.label2 = tk.Label(self)
        self.label2.grid(row=3, column=2, rowspan=4, sticky="nsew")
        # 撮影画像を表示するcanvas
        self.label3 = tk.Label(self)
        self.label3.grid(row=7, column=1, rowspan=4, sticky="nsew")
        self.label4 = tk.Label(self)
        self.label4.grid(row=7, column=2, rowspan=4, sticky="nsew")


    def load_device_list(self):
        device_list =[]
        device_list = ic4.DeviceEnum.devices()

        if len(device_list) == 0:
            print("No devices found")
            
        return device_list

    def load_devices(self):
        try:
            self.devices =  self.load_device_list()
                
            if llen(self.devices) == 0 :
                self.comboBoxDevices1["values"] = ["None"]
                self.comboBoxDevices2["values"] = ["None"]
                self.comboBoxDevices1.current(0)
                self.comboBoxDevices2.current(0)
                self.buttonConnect.config(state=tk.DISABLED)
                return

            self.comboBoxDevices1["values"] = self.devices
            self.comboBoxDevices2["values"] = self.devices
            if self.devices:
                self.comboBoxDevices1.current(0)
                self.comboBoxDevices2.current(1 if len(self.devices) > 1 else 0)

        except Exception as e:
            print(f"Error loading devices: {e}")

     def ic4_connect_cameras(self):
        self.left_grabber.device_open(self.comboBoxDevices1.get())
        self.right_grabber.device_open(self.comboBoxDevices2.get())

        if  self.left_grabber.is_device_open and self.right_grabber.is_device_open:
            self.left_grabber.stream_setup(self.left_sink , setup_option=ic4.StreamSetupOption.ACQUISITION_START)
            self.right_grabberr.stream_setup(self.right_sink , setup_option=ic4.StreamSetupOption.ACQUISITION_START)
            self.update_frame()
        else:
            print("カメラ接続できません")

    def connect_modbus_server(self):
        if not self.devices:
            print("No cameras available to start.")
            return

        if not self.client:
            try:

                self.client = ModbusClient( self.modbus_server_ip , port= self.modbus_server_port) 
                if not self.client.connect():
                    print("Failed to connect to Modbus server")
                    return
                print("Connected to Modbus server")
                self.read_robot_register()
             
            except Exception as e:
                print(f"Error connecting to Modbus server or cameras: {e}")

    def update_frame(self):
        try:
            left_image = self.left_sink.snap_single(1000)
            right_image = self.left_sink.snap_single(1000)

            img1 = ImageTk.PhotoImage(image=Image.fromarray(left_image.numpy_wrap()))
            self.label1.imgtk = img1
            self.label1.configure(image=img1)

            img2 = ImageTk.PhotoImage(image=Image.fromarray(right_image.numpy_wrap()))
            self.label2.imgtk = img2
            self.label2.configure(image=img2)

        except Exception as e:
            print(f"フレームの更新エラー: {e}")

        self.after(self.update_interval, self.update_frame)

    def capture_images(self):
       try:
            left_image = self.left_sink.snap_single(1000)
            right_image = self.left_sink.snap_single(1000)

            img1 = ImageTk.PhotoImage(image=Image.fromarray(left_image.numpy_wrap()))
            self.label3.imgtk = img1
            self.label3.configure(image=img1)

            img2 = ImageTk.PhotoImage(image=Image.fromarray(right_image.numpy_wrap()))
            self.label4.imgtk = img2
            self.label4.configure(image=img2)
                
        except Exception as e:
            print(f"Error capturing images: {e}")
            
    def send_complete_flag(self):
        #処理完了フラグの送信
        try:
            if self.client:
                self.client.write_coil(self.complete_flag_address, True)
                print("Complete flag sent")
        except Exception as e:
            print(f"Error sending complete flag: {e}")
            
    def send_pose_data(self,data_list):
        #物体座標データの送信
        def make_send_data(data_list):
            send_data=[]
            for data in data_list:
                byte=struct.pack('>f', data)
                data_int  = struct.unpack('>HH',byte)
                send_data.append(data_int[0])
                send_data.append(data_int[1])
            return send_data
        
        try:
            if self.client:
                data =make_send_data(data_list)
                self.client.write_registers(self.register_address, data)
                print("pose data sent")
        except Exception as e:
            print(f"Error sending pose data: {e}")
    
    def read_robot_register(self):

        try:
            result_di8 = self.client.read_discrete_inputs(self.capture_flag_address,2).bits[0]
           
            if result_di8:  # DO8 is on
                if self.flag_is_checked:
                    pass
                else:
                    print("DO8 is on")
                    # カメラからの画像キャプチャ
                    self.capture_images()
                    # 座標データを送信
                    self.send_pose_data(self.dummy_data)     
                    # 処理完了フラグを送信
                    self.send_complete_flag()
                    self.flag_is_checked = True
            else:
                print("D08 is off")
                self.flag_is_checked = False

            self.after(self.check_interval, self.read_robot_register)

        except Exception as e:
            print(f"Modbusのエラーチェック: {e}")

    def on_closing(self):
            # grabber.stream_stop()
            # grabber.device_close()
        if self.left_grabber.is_device_open:
            if self.left_grabber.is_streaming:
                self.left_grabber.stream_stop()
            self.left.device_close

        if self.right_grabber.is_device_open:
            if self.right_grabber.is_streaming:
                self.right_grabber.stream_stop()
            self.right.device_close()
   
        if self.client:
            self.client.close()
        print("app closed")
        self.destroy()

if __name__ == "__main__":
    
    app = CameraApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()


