import json
import threading
import paho.mqtt.client as mqtt



MQTT_BROKER = "9d91117459c94f99b527b4d80d9fed5f.s1.eu.hivemq.cloud"
MQTT_PORT = 8883

MQTT_USERNAME = "MMATTO"
MQTT_PASSWORD = "MQTTpassword"
MQTT_TOPIC = "stress/readings"

#Get the readings
latest_data = {
    "HR": None,
    "BVP": None,
    #"ECG": None
}

#auto connects to the client
def on_connect(client, userdata, flags, rc):
    if rc==0:
        print(" You are connected to the broker, Whoop WHoop!")
        client.subscribe(MQTT_TOPIC)
    else:
        print("Not able to connect to the MQTT")

def on_message (client, userdata, msg):
    try: #convert to json what it receives from the MQTT
        data = json.loads(msg.payload.decode())


        HR = float(data["HR"])
        BVP = float (data["BVP"])
        #ECG = float (data["ECG"])
    except Exception as e:
        print("There is an error with receiving the data from the MQTT", e)

        print("HR:", HR)
        print("BVP:", BVP)
       # print("ECG:", ECG)


mqtt_client = mqtt.Client()

#TLS for encryption
mqtt_client.tls_set()

#login for the mqtt
mqtt_client.username_pw_set(
    MQTT_USERNAME,
    MQTT_PASSWORD
)

#choose the correct function
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message

#60 for every 60 seconds to make the connection
if __name__ == "__main__":
    mqtt_client.connect(
        MQTT_BROKER,
        MQTT_PORT,
        60
)
#keeps the mqtt client running so that it does not stop
    mqtt_client.loop_forever()
