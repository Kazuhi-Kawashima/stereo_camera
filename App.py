import tkinter as tk
from tkinter import ttk, font
from datetime import datetime
import struct

import numpy as np
import cv2
import visiontransfer
from PIL import Image, ImageTk
from pymodbus.client import ModbusTcpClient as ModbusClient

import MegaposeEstimater
import vis_util

#test
#test2
class CameraApp(tk.Tk):
    def __init__(self):
        self.transfer = None
        self.rec3d = None
        
        self.megapose = MegaposeEstimater.MegaposeEstimater(MegaposeEstimater.LOCAL_DIR / "data", True)
        self.K = np.array([[1174.4691, 0.0, 702.0466], [0.0, 1176.8887, 482.63185], [0.0, 0.0, 1.0]])
       
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
        #self.grid_rowconfigure(index=10, weight=1)
        #self.grid_rowconfigure(index=11, weight=1)
        #self.grid_rowconfigure(index=12, weight=1)
    
        self.grid_columnconfigure(index=0, weight=2)
        self.grid_columnconfigure(index=1, weight=4)
        self.grid_columnconfigure(index=2, weight=4)

        self.capture_flag_address = 15 #画像キャプチャフラグのアドレス
        self.complete_flag_address = 50 #完了フラグのアドレス
        self.absense_flag_address = 52 #フラグのアドレス
        self.register_address = 132 #物体座標送信用のレジスタのアドレス始点
        self.pose_data=[]
        self.update_interval = 20 # 画像更新間隔
        self.check_interval = 500 # DO8のチェック間隔
        self.modbus_server_ip ='192.168.3.41' #ModbusサーバーのIPアドレス
        self.modbus_server_port =502 #Modbusサーバーのポート番号

        self.flag_is_checked =False
        self.devices = []
        self.client = None  

        self.create_widgets()
        self.load_devices()      

    def create_widgets(self):
        "ウィジェットの作成"
        # カメラ選択用のコンボボックス
        tk.Label(self, text="device").grid(row=0, column=1, padx=10, pady=5, sticky='w')
        self.comboBoxDevices1 = ttk.Combobox(self, width=20)
        self.comboBoxDevices1.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        # Mカメラ接続用の接続ボタン
        self.camera_connect_btn = tk.Button(self, text="camera connect", command=self.connect_nerian_camera)
        self.camera_connect_btn.grid(row=2, column=0, padx=5, pady=5, sticky="ew")

        # Modbusサーバー接続用の接続ボタン
        self.buttonConnect = tk.Button(self, text="Connect", command=self.connect_modbus_server)
        self.buttonConnect.grid(row=3, column=0, padx=5, pady=5, sticky="ew")
        # capture btn 
        #self.buttonConnect = tk.Button(self, text="capture", command=self.capture_images)
        #self.buttonConnect.grid(row=4, column=0, padx=5, pady=5, sticky="ew")

        # カメラ映像を表示するcanvas
        self.label1 = tk.Label(self)
        self.label1.grid(row=3, column=1, rowspan=4, sticky="nsew")
        self.label2 = tk.Label(self)
        self.label2.grid(row=3, column=2, rowspan=4, sticky="nsew")
        # 撮影画像を表示するcanvas
        self.label3 = tk.Label(self)
        self.label3.grid(row=7, column=1, rowspan=4, sticky="ew")
        #self.label4 = tk.Label(self)
        #self.label4.grid(row=7, column=2, rowspan=4, sticky="nsew")
        
        font_obj = font.Font(size=28)
        self.label5 = tk.Label(self, text="x:mm",font=font_obj)
        self.label5.grid(row=4, column=0, sticky="nsew")
        self.label6 = tk.Label(self, text="y:mm",font=font_obj)
        self.label6.grid(row=5, column=0, sticky="nsew")
        self.label7 = tk.Label(self, text="z:mm",font=font_obj)
        self.label7.grid(row=6, column=0, sticky="nsew")
        self.label8 = tk.Label(self, text="xr:°",font=font_obj)
        self.label8.grid(row=7, column=0, sticky="nsew")
        self.label9 = tk.Label(self, text="yr:°",font=font_obj)
        self.label9.grid(row=8, column=0, sticky="nsew")
        self.label10 = tk.Label(self, text="zr:°",font=font_obj)
        self.label10.grid(row=9, column=0, sticky="nsew")
        
    def load_device_list(self):
        device_list =[]
        device_enum = visiontransfer.DeviceEnumeration()
        device_list = device_enum.discover_devices()

        if len(device_list) < 1:
            print('No devices found')
            
        return device_list

    def load_devices(self):

        try:
            self.devices =  self.load_device_list()
                
            if len(self.devices) == 0 :
                self.comboBoxDevices1["values"] = ["None"]
                self.comboBoxDevices1.current(0)
                self.buttonConnect.config(state=tk.DISABLED)
                return
	    
            self.comboBoxDevices1["values"] = self.devices

            if self.devices:
                self.comboBoxDevices1.current(0)

        except Exception as e:
            print(f"Error loading devices: {e}")

    def connect_nerian_camera(self):
        device =self.devices[0]
        params = visiontransfer.DeviceParameters(device)
        params.set_operation_mode(visiontransfer.OperationMode.STEREO_MATCHING)

        try:
            self.transfer = visiontransfer.AsyncTransfer(device)
            self.rec3d = visiontransfer.Reconstruct3D()
            print(self.transfer.is_connected())
            self.update_frame()
        except:
            print("cant connect camera")

    def connect_modbus_server(self):

        #if not self.devices:
         #  print("No cameras available to start.")
          # return

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
            image_set = self.transfer.collect_received_image_set()
            
            #rgbd = self.rec3d.create_open3d_rgbd_image(image_set, depth_trunc=600, color_source=3,depth_scale=0.001,convert_rgb_to_intensity=False, convert_intensity_to_rgb=True)
            rgbd = self.rec3d.create_open3d_rgbd_image(image_set, depth_trunc=600,depth_scale=0.001,color_source=2,convert_rgb_to_intensity=False, convert_intensity_to_rgb=True)
            color= np.array((rgbd.color), dtype=np.uint8)
            depth = np.array(rgbd.depth)  
            
            x_min = np.min(depth)
            x_max = np.max(depth)
            depth_scaled = (depth - x_min) / (x_max - x_min)
            color = cv2.resize(color,(640,480))
            depth_scaled = cv2.resize(depth_scaled,(640,480))*255
            
            img1 = ImageTk.PhotoImage(image=Image.fromarray(color))
            self.label1.imgtk = img1
            self.label1.configure(image=img1)

            img2 = ImageTk.PhotoImage(image=Image.fromarray(depth_scaled))
            self.label2.imgtk = img2
            self.label2.configure(image=img2)

        except Exception as e:
            print(f"frame update error: {e}")
        finally:
            self.after(self.update_interval, self.update_frame)

    def capture_images(self):
        
        try:
            image_set = self.transfer.collect_received_image_set()
            #rgbd = self.rec3d.create_open3d_rgbd_image(image_set, depth_trunc=600, color_source=3,depth_scale=0.001,convert_rgb_to_intensity=False, convert_intensity_to_rgb=True)
            rgbd = self.rec3d.create_open3d_rgbd_image(image_set, depth_trunc=600,depth_scale=0.001,color_source=2,convert_rgb_to_intensity=False, convert_intensity_to_rgb=True)
            color= np.array((rgbd.color), dtype=np.uint8)
            depth = np.array((rgbd.depth),dtype=np.float32)  

            outputs,boxes= self.megapose.run_inference(color,depth)
            #outputs=[[[0.422618,0,0,0.906308],[0,400,0]]]
            self.pose_data = []
            if outputs:  
                print(np.array(outputs[0][0]))
                print(np.array(outputs[0][1]))    
                self.pose_data = vis_util.create_pose_data(np.array(outputs[0][0]),np.array(outputs[0][1]))
                
                color = vis_util.draw_boxes(color,boxes)
                color = vis_util.draw_axis_from_qua(color,np.array(outputs[0][0]),np.array(outputs[0][1]),self.K)
                color = cv2.resize(color,(640,480))
                
                img1 = ImageTk.PhotoImage(image=Image.fromarray(color))
                self.label3.imgtk = img1
                self.label3.configure(image=img1)
            
        except Exception as e:
            print(f"Error capturing images: {e}")
    
    def send_flag(self,address,flag:bool):
        #フラグの送信
        try:

            if self.client:
                self.client.write_coil(address, flag)
                print("flag sent")

        except Exception as e:
            print(f"Error sending flag: {e}")
            
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
                data = make_send_data(data_list)
                print(data)
                self.client.write_registers(self.register_address, data)
                print("pose data sent")
                
                self.label5['text']=f'x:{data_list[0]:.3f}mm'
                self.label6['text']=f'y:{data_list[1]:.3f}mm'
                self.label7['text']=f'z:{data_list[2]:.3f}mm'
                self.label8['text']=f'xr:{data_list[3]:.3f}°'
                self.label9['text']=f'yr:{data_list[4]:.3f}°'
                self.label10['text']=f'zr:{data_list[5]:.3f}°'
                
        except Exception as e:
            print(f"Error sending pose data: {e}")
    
    def read_robot_register(self):

        try:
            result_di8 = self.client.read_discrete_inputs(self.capture_flag_address,2).bits[0]
           
            if result_di8:  # DO8 is on
                
                if self.flag_is_checked:
                    pass
                else:
                    self.send_flag(self.absense_flag_address,False)
                    self.send_flag(self.complete_flag_address,False)
                    print("DO8 is on")
                    # 画像キャプチャ
                    self.capture_images()
                    send_data=self.pose_data

                    if send_data:
                        # 座標データを送信  
                        self.send_pose_data(send_data)    
                        # 処理完了フラグを送信
                        self.send_flag(self.complete_flag_address,True)
                        self.flag_is_checked = True
                    else:
                        self.send_flag(self.absense_flag_address,True)

            else:
                if self.flag_is_checked:
                    print("D08 is off")
                    self.flag_is_checked = False

        except Exception as e:
            print(f"Modbus error: {e}")

        finally:    
            self.after(self.check_interval, self.read_robot_register)

    def on_closing(self):
    
        if self.client:
            self.client.close()
        self.megapose = None
        print("app closed")
        self.destroy()

if __name__ == "__main__":
    
    app = CameraApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
