from flask import Flask, render_template, redirect, url_for
from multiprocessing import Process
import subprocess
import os
from datetime import datetime
import time
import threading
import serial


app = Flask(__name__)
app.config['STATUS'] = 'OFF'
app.config['CAMERA_WORKERS'] = []
app.config['SPECTROMETER_WORKERS'] = []
app.config['PORT'] = 8080

# Placeholder classes for spectrometers and trigger

class Spectrometer:
    def __init__(self):
        # Add your spectrometer initialization code here
        pass

    def setup_worker(self):
        # Add your spectrometer worker setup code here
        pass

    def start_worker(self):
        # Start the spectrometer worker here
        pass

    def stop_worker(self):
        # Stop the spectrometer worker here
        pass


class Trigger:
    def __init__(self):
        self.initiated = datetime.now()
        self.last_time_triggered = datetime.now()
        self.triggered = False
        self.running = False
        self.connected = False
        self.time_to_clear = False

        # self.arduino_port = '/dev/ttyACM1'
        self.arduino_port = '/dev/myArduino1'
        # self.arduino_port = '/dev/myArduino2'
        # self.arduino_port = 'ttyUSB0'

        self.arduino_baudrate = 1000000

        self.connect()
        self.worker = threading.Thread(target=self.communicate, args=())

        # self.arduino = serial.Serial(port=self.arduino_port, baudrate=self.arduino_baudrate)

        # self.
    def connect(self):
        try:
            self.arduino = serial.Serial(port=self.arduino_port, baudrate=self.arduino_baudrate)
            self.connected = True
            print('Arduino connected')
            return True
        except:
            print("couldn't initiate arduino ")
            self.connected = False
            return False

    def communicate(self):
        print("Communication attempted")
        while True:
            if not self.connected:
                print("During communication with arduino it was not able to connect, trying to reconnect")
                time.sleep(1)
                self.connect()
                continue
            try:
                data = self.arduino.readline().decode().strip()
                print(f"Arduino said {data}")
                if data == 'X':
                    self.triggered = True
                    self.last_time_triggered = datetime.now()

                if data == 'C':
                    self.time_to_clear = True
                time.sleep(0.001)
            except Exception as e:
                print(f"Communication error: {e}")
                self.connected = False
                time.sleep(1)
            # print(f"{self.triggered}_{datetime.now().strftime("%Y%m%d-%H%M%S")}")


def find_cameras():
    """Finds USB cameras with vendor and model IDs '0c45:636d'."""
    devices = []
    for dev in os.listdir('/dev'):
        if dev.startswith('video'):
            device_path = f"/dev/{dev}"
            vendor_cmd = f"udevadm info --query=all --name={device_path} | grep 'ID_VENDOR_ID=0c45'"
            model_cmd = f"udevadm info --query=all --name={device_path} | grep 'ID_MODEL_ID=636d'"
            vendor_match = subprocess.run(vendor_cmd, shell=True, capture_output=True)
            model_match = subprocess.run(model_cmd, shell=True, capture_output=True)
            if vendor_match.returncode == 0 and model_match.returncode == 0:
                devices.append(device_path)
                print(device_path)

    print(f"Devices {devices}")

    return devices


def setup_camera_worker(device):
    """Sets up and starts the camera configuration worker."""
    subprocess.run(f"v4l2-ctl -d {device} --set-fmt-video=width=640,height=480,pixelformat=MJPG", shell=True)
    subprocess.run(f"v4l2-ctl --device={device} --set-parm=120", shell=True)


def start_cameras():
    """Starts camera workers for each connected camera."""
    # cameras = find_cameras()
    cameras = ["/dev/video8"]
    workers = []
    for cam in cameras:
        worker = Process(target=setup_camera_worker, args=(cam,))
        worker.start()
        workers.append(worker)
    app.config['CAMERA_WORKERS'] = workers


def stop_cameras():
    """Stops all camera workers."""
    for worker in app.config['CAMERA_WORKERS']:
        worker.terminate()
    app.config['CAMERA_WORKERS'] = []


@app.route('/')
def index():
    return render_template('index.html', status=app.config['STATUS'])


@app.route('/toggle/<action>')
def toggle(action):
    if action == 'on' and app.config['STATUS'] == 'OFF':
        app.config['STATUS'] = 'ON'
        # Start camera workers
        start_cameras()
        # Start spectrometer workers
        for spectrometer in spectrometers:
            spectrometer.setup_worker()
            spectrometer.start_worker()
        # Here, you could start monitoring the trigger if needed

    elif action == 'off' and app.config['STATUS'] == 'ON':
        app.config['STATUS'] = 'OFF'
        # Stop camera workers
        stop_cameras()
        # Stop spectrometer workers
        for spectrometer in spectrometers:
            spectrometer.stop_worker()

    return redirect(url_for('index'))


if __name__ == '__main__':
    # Initialize spectrometers and trigger
    spectrometers = [Spectrometer(), Spectrometer()]  # Assuming 2 spectrometers


    # app.run(host='0.0.0.0', port=app.config['PORT'])
    print(f"Nor for arduino")
    trigger = Trigger()
    trigger.worker.start()

    web_app_worker = threading.Thread(target=app.run(host='0.0.0.0', port=app.config['PORT']), args=())
    web_app_worker.start()
