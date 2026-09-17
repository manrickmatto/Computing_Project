import json
import threading
import serial
import time

#location of where the USB is/ BAUD = transmission rate
Port = "COM3"
Baud_rate = 115200

#Get the readings
latest_data = {
    "hr": None,
    "bvp": None,
}
#keep the readings for BVP and HR seperate
data_lock = threading.Lock()

#set up the connection for the application to find the sensor
def read_sensor():
    #keep the dictionary updating but not changing the values
    global latest_data
    try:
        with serial.Serial(
            port=Port,
            baudrate=Baud_rate,
            timeout=1
        ) as sensor:

            print (f"The sensor is connected to {Port}")
            print("Port open:", sensor.is_open)
            #pause the code for the device to start sending readings after being detected
            time.sleep(2)

            #read the raw data coming from the sensor
            while sensor.is_open:
                line =sensor.readline().decode(
                        "utf-8",
                        errors="ignore"
                    ).strip() #remove null values/ blank values

                if not line:
                    continue

                try:
                    #convert to JSON
                    data=json.loads(line)

                    if "hr" in data and "bvp" in data:
                        with data_lock:
                            latest_data.update({
                                "hr": data["hr"],
                                "bvp": data["bvp"]
                            })
                        print(latest_data)

                except json.JSONDecodeError:
                    print("Invalid data", repr(line))
    
    except serial.SerialException as error:
        print(f"Serial error: {error}")
            
def start_sensor():
    thread = threading.Thread(
        target=read_sensor,
        daemon=True
    )
    thread.start()

def get_latest_data():
    with data_lock:
        return latest_data.copy()