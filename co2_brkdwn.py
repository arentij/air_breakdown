from flask import Flask
import subprocess
import cv2
from datetime import datetime
import serial
import datetime
import threading
import time
import os
import seabreeze
import usb.core
seabreeze.use('pyseabreeze')
from seabreeze.spectrometers import Spectrometer, list_devices
import pandas as pd
import numpy
import csv
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import matplotlib


matplotlib.use("Agg")

app = Flask(__name__)
app.config['PORT'] = 8080

usb_cam_dev = "/dev/video6"
usb_cam_dev2 = "/dev/video14"
# Create output directory if it doesn't exist
output_dir = "air_breakdown"
os.makedirs(output_dir, exist_ok=True)


class USB_spectrometer:
    def __init__(self, integ_time=3000, max_time=1, dsc=0, n=0, save_folder=output_dir, serial="USB2G410"):
        self.running = False
        self.connected = False
        self.triggered = False
        self.max_time = max_time

        self.need_to_wait = True
        self.skip = True

        self.previous_time = datetime.datetime.now()
        self.triggered_time = datetime.datetime.now()
        self.dark_time = datetime.datetime.now()

        self.dark_intensities = None

        self.previous_int = None
        self.trigger_int = None
        self.dark_int = None

        self.last_time_triggered = datetime.datetime.now()

        self.serial = serial # old "USB2G410", new "SR200584"
        self.dsc = dsc
        self.N_exp = n
        self.save_folder = save_folder

        self.index = 0
        self.created = datetime.datetime.now()
        self.accessed = datetime.datetime.now()

        self.time_created = datetime.datetime.now()

        self.devices = []
        self.devices_found = False

        before = datetime.datetime.now()
        self.search_device_worker = threading.Thread(target=self.search_device, args=())
        after = datetime.datetime.now()
        # print(f"Time to create device worker {(after-before).total_seconds()}")
        self.integration_time = integ_time

        self.worker = threading.Thread()

        self.waves = []
        self.spectra = []
        self.times = []

        self.connect = threading.Thread(target=self.connecting, args=())

    def search_device(self):
        print("Launching Spectrometer Device search")
        try:
            self.devices = list_devices()
            self.devices_found = True
            print(f"Found {self.devices}")
        except Exception as e:
            print("couldn't find any device")
            self.devices_found = False
        print("Spectrometer search_device successful")
        return True

    def connecting(self):
        print(f"Attempting connecting to the spectrometer {self.serial}")
        if True:

            try:
                self.spect = Spectrometer.from_serial_number(self.serial)
                # self.spect = Spectrometer.from_serial_number("SR200584")
                self.time_connected = datetime.datetime.now()
                self.spect.integration_time_micros(self.integration_time)

                self.times = datetime.datetime.now()
                self.waves = self.spect.wavelengths()
                self.inten = self.spect.intensities()

                self.connected = True

            except Exception as e:
                print(f"Could not connect to the spectrometer {self.serial} due to: {e}")
                self.connected = False

                time.sleep(0.1)

    def running_reading_listening(self):
        if not self.connected:
            self.connecting()
        time_started_running = datetime.datetime.now()
        time.sleep(0.15)

        while True:
            try:
                self.triggered_time = datetime.datetime.now()
                self.inten_exp = self.spect.intensities()
                self.waves = self.spect.wavelengths()
                time.sleep(0.5)
                break
            except AttributeError as e:
                print(f"spectrometer said {e}")
        print(f"spectrometer {self.serial} is running")

        leaving = False
        while True:
            self.previous_time = self.triggered_time
            self.previous_int = self.inten_exp.copy()

            self.triggered_time = datetime.datetime.now()
            self.inten_exp = self.spect.intensities()

            if trigger.triggered:
                if True or self.last_time_triggered != trigger.last_time_triggered:

                    self.triggered = False
                    while self.need_to_wait:
                        time.sleep(0.00001)

                    # time.sleep(0.001)
                    if not self.skip:
                        print(f"Read spectras on {self.serial}, time to write")
                        self.dark_time = datetime.datetime.now()
                        self.dark_intensities = self.spect.intensities()
                        self.save_data()

                        # trigger.status_to_write = 0
                    else:
                        self.skip = True

            if leaving:
                break

    def save_data(self):
        # Format time from trigger for filename

        time_str = trigger.last_time_triggered.strftime('%Y-%m-%d_%H-%M-%S.%f')[:]  # up to us precision

        filename = f"spec_{self.serial}_{time_str[:-3]}.csv"
        # filename_info = f"spec_{self.serial}_{time_str[:-3]}_info.csv"
        filepath = os.path.join(output_dir, filename)
        os.makedirs(output_dir, exist_ok=True)

        time_inten = (self.triggered_time - trigger.last_time_triggered).total_seconds() * 1000  # up to us precision
        time_dark = (self.dark_time - trigger.last_time_triggered).total_seconds() * 1000  # up to us precision
        time_prev = (self.previous_time - trigger.last_time_triggered).total_seconds() * 1000  # up to us precision
        # Write wavelengths and intensities to CSV
        with open(filepath, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Wavelength (nm)", "Intensity", "Dark intensity", "Previous intensity"])  # Header row
            writer.writerow([time_str, time_inten, time_dark, time_prev])  # Header row
            for wavelength, intensity, dark_int, old_int in zip(self.waves, self.inten_exp, self.dark_intensities, self.previous_int ):
                writer.writerow([wavelength, intensity, dark_int, old_int])

        print(f"Data saved to {filepath}")

        self.last_time_triggered = trigger.last_time_triggered

    def setup_worker(self, dsc=0, n=0, save_folder=output_dir):

        self.dsc = dsc
        self.N_exp = n
        self.save_folder = save_folder
        self.triggered = False
        self.worker = threading.Thread(target=self.running_reading_listening, args=())
        return True


class Trigger:
    def __init__(self):
        self.initiated = datetime.datetime.now()
        self.last_time_triggered = datetime.datetime.now()
        self.triggered = False
        self.running = False
        self.connected = False
        self.time_to_clear = False
        self.status_to_write = 0
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
                # print(f"Arduino said {data}")
                if data == 'X':
                    self.triggered = True
                    self.last_time_triggered = datetime.datetime.now()
                    time.sleep((0.01))
                    # print(f"Trigger on")

                if data == 'C':
                    self.time_to_clear = True
                if data == 'O':
                    time.sleep((0.1))
                    self.triggered = False
                    # print(f"Trigger off")

            except Exception as e:
                print(f"Communication error: {e}")
                self.connected = False
                time.sleep(1)
            # print(f"{self.triggered}_{datetime.now().strftime("%Y%m%d-%H%M%S")}")


def configure_camera():
    """Configures the USB camera on /dev/video8."""
    subprocess.run(f"v4l2-ctl -d {usb_cam_dev} --set-fmt-video=width=640,height=480,pixelformat=MJPG", shell=True)
    subprocess.run(f"v4l2-ctl --device={usb_cam_dev} --set-parm=120", shell=True)
    subprocess.run(f"v4l2-ctl --device={usb_cam_dev} -c auto_exposure=1", shell=True)
    subprocess.run(f"v4l2-ctl --device={usb_cam_dev} -c exposure_time_absolute=10", shell=True)

    subprocess.run(f"v4l2-ctl -d {usb_cam_dev2} --set-fmt-video=width=800,height=600,pixelformat=MJPG", shell=True)
    subprocess.run(f"v4l2-ctl --device={usb_cam_dev2} --set-parm=90", shell=True)
    subprocess.run(f"v4l2-ctl --device={usb_cam_dev2} -c auto_exposure=1", shell=True)
    subprocess.run(f"v4l2-ctl --device={usb_cam_dev2} -c exposure_time_absolute=1000", shell=True)


def camera_worker(trigger):
    """Starts the camera worker to capture frames and save them on trigger."""
    configure_camera()  # Set the camera configuration
    last_time_cam_triggered = datetime.datetime.now()
    # status_to_write = 0
    # Open video capture for the specified device
    cap = cv2.VideoCapture(usb_cam_dev2)
    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    try:
        while True:
            # Read a frame from the camera
            # for spectrometer in spectrometers:
            #     spectrometer.need_to_wait = False
            #     spectrometer.skip = True
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image.")
                break

            # Check if trigger is activated
            if trigger.triggered:
                trigger.status_to_write = 0
                for spectrometer in spectrometers:
                    spectrometer.need_to_wait = True
                    spectrometer.skip = False
                # Reset the trigger after capturing the frame
                # trigger.triggered = False
                if last_time_cam_triggered != trigger.last_time_triggered:
                    # Save the current frame with timestamp

                    max_pixel = np.max(frame)
                    min_pixel = np.min(frame)

                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # Calculate the average intensity
                    average_intensity = np.mean(gray_frame)

                    print(f"max pixel is {max_pixel}, min pixel is {min_pixel}, average intensity={int(average_intensity)}")
                    # print()
                    if max_pixel >= pixel_threshold and average_intensity > average_threshold:
                        for spectrometer in spectrometers:
                            spectrometer.need_to_wait = False
                            spectrometer.skip = False
                            print(f"This is a proper light level: {int(average_intensity)}")
                        trigger.status_to_write = 1
                        timestamp = trigger.last_time_triggered.strftime('%Y-%m-%d_%H-%M-%S.%f')[:-3]
                        filename = os.path.join(output_dir, f"frame_{timestamp}.jpg")
                        cv2.imwrite(filename, frame)
                        print(f"Saved frame to {filename}")
                        last_time_cam_triggered = trigger.last_time_triggered
                    elif max_pixel < pixel_threshold:
                        for spectrometer in spectrometers:
                            spectrometer.need_to_wait = False
                            # print(f"Not enought light!")
                            spectrometer.skip = True
                # time.sleep(0.2)
    finally:
        # Release the camera resource when done
        cap.release()


@app.route('/start')
def start_app():
    # Start trigger monitoring in its own thread
    # trigger_thread = threading.Thread(target=trigger.start)
    # trigger_thread.daemon = True
    # trigger_thread.start()

    # Start camera worker in a separate thread
    camera_thread = threading.Thread(target=camera_worker, args=(trigger,))
    camera_thread.daemon = True
    camera_thread.start()

    return "Application started with camera and trigger worker."


if __name__ == '__main__':
    # Initialize the trigger
    # print(seabreeze.spectrometers.list_devices())

    pixel_threshold = 250
    average_threshold = 40

    trigger = Trigger()
    trigger.worker.start()
    trigger.status_to_write = 0  # 0 to wait, 1 to write, -1 to skip

    camera_thread = threading.Thread(target=camera_worker, args=(trigger,))
    camera_thread.daemon = True
    camera_thread.start()

    integration_time_USB2G410 = 13000
    usb_spec2000 = USB_spectrometer(integ_time=integration_time_USB2G410, serial="USB2G410", max_time=1)

    integration_time_SR200584 = 13000
    usb_specSR2 = USB_spectrometer(integ_time=integration_time_SR200584, serial="SR200584", max_time=1)

    spectrometers = [usb_specSR2, usb_spec2000]
    # spectrometers = [usb_specSR2]

    for spectrometer in spectrometers:
        spectrometer.setup_worker()
        spectrometer.worker.start()
        time.sleep(0.1)

    folder_to_monitor = output_dir

    # monitor = CSVMonitor(folder_to_monitor)
    # monitor.start()

    app.run(host='0.0.0.0', port=app.config['PORT'])
