import flwr as fl
import tensorflow as tf
from tensorflow import keras

import os
from urllib.request import urlopen
from urllib.error import URLError, HTTPError
import requests
from fastapi import FastAPI, BackgroundTasks
import asyncio
import uvicorn
from pydantic.main import BaseModel
import logging
import json
import boto3
from functools import partial
from flwr.client import NumPyClient
import time

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

app = FastAPI()


class FLclient_status(BaseModel):
    FL_client_online: bool = True
    FLCLstart: bool = False
    FLCFail: bool = False
    FL_server_IP: str = None  # '10.152.183.181:8080'


status = FLclient_status()


def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# Define Flower client
class CustomeClient:
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    #def get_properties(self, **kwargs):
    #    super().get_properties(self)

    def get_parameters(self):
        """Get parameters of the local model."""
        return self.model.get_weights()
        # raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        # Train the model using hyperparameters from config
        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size,
            epochs,
            validation_split=0.1,
        )

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
        }
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, 32, steps=steps)
        num_examples_test = len(self.x_test)
        return loss, num_examples_test, {"accuracy": accuracy}


# Load CIFAR-10 dataset


@app.on_event("startup")
def startup():
    pass
    # loop = asyncio.get_event_loop()
    # loop.set_debug(True)
    # loop.create_task(run_client())


@app.get("/start/{Server_IP}")
async def flclientstart(background_tasks: BackgroundTasks, Server_IP: str):
    global status
    global model
    model = build_model()

    print('start')
    status.FLCLstart = True
    status.FL_server_IP = Server_IP
    background_tasks.add_task(run_client)
    return status


@app.get('/test')
def get_model_test():
    if status.FLCLstart == False:
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        print(model.evaluate(x_test, y_test))


@app.get('/online')
def get_info():
    return status


async def run_client():
    global model
    try:
        # time.sleep(10)
        model.load_weights('/model/model.h5')
        pass
    except Exception as e:
        print('[E][PC0001] learning', e)
        status.FLCFail = True
        await notify_fail()
        status.FLCFail = False
    await flower_client_start()


async def flower_client_start():
    print('learning')
    global status
    global model
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    try:
        loop = asyncio.get_event_loop()
        client = CustomeClient(model, x_train, y_train, x_test, y_test)
#        assert type(client).get_properties == fl.client.NumPyClient.get_properties
        print(status.FL_server_IP)
        #fl.client.start_numpy_client(server_address=status.FL_server_IP, client=client)
        request = partial(fl.client.start_numpy_client, server_address=status.FL_server_IP, client=client)
        await loop.run_in_executor(None, request)
        
        await model_save()
        del client
    except Exception as e:

        print('[E][PC0002] learning', e)
        status.FLCFail = True
        await notify_fail()
        status.FLCFail = False
        # raise e


async def model_save():
    print('model_save')
    global model
    try:
        model.save('/model/model.h5')
        await notify_fin()
        model=None
    except Exception as e:
        print('[E][PC0003] learning', e)
        status.FLCFail = True
        await notify_fail()
        status.FLCFail = False


async def notify_fin():
    global status
    status.FLCLstart = False
    loop = asyncio.get_event_loop()
    future2 = loop.run_in_executor(None, requests.get, 'http://localhost:8080/trainFin')
    r = await future2
    print('try notify_fin')
    if r.status_code == 200:
        print('trainFin')
    else:
        print(r.content)


async def notify_fail():
    global status
    status.FLCLstart = False
    loop = asyncio.get_event_loop()
    future1 = loop.run_in_executor(None, requests.get, 'http://localhost:8080/trainFail')
    r = await future1
    print('try notify_fail')
    if r.status_code == 200:
        print('trainFin')
    else:
        print(r.content)


# Start Flower client
# s3에 model 없고 환경변수를 탐색하여 ENV가 init이라면 s3에 초기 가중치를 업로드 한다.
from botocore.exceptions import ClientError


def S3_check(s3_client, bucket, key):  # 없으면 참
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
    except ClientError as e:
        return int(e.response['Error']['Code']) != 404
    return False


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    print(tf.version.VERSION)

#     if os.environ.get('ENV', 'development') == 'init':
    if os.environ.get('ENV') is not None:
        res = requests.get('http://10.152.183.18:8000' + '/FLSe/info')  # 서버측 manager
        S3_info = res.json()['Server_Status']
        model = build_model()
        model.save(S3_info['S3_key'])
        ##########서버에 secret피일 이미 있음 #################
        ACCESS_KEY_ID = os.environ.get('ACCESS_KEY_ID')
        ACCESS_SECRET_KEY = os.environ.get('ACCESS_SECRET_KEY')
        BUCKET_NAME = os.environ.get('BUCKET_NAME')
        s3_client = boto3.client('s3', aws_access_key_id=ACCESS_KEY_ID,
                                 aws_secret_access_key=ACCESS_SECRET_KEY)
        if S3_check(s3_client, BUCKET_NAME, S3_info['S3_key']):
            print('모델 없음')
            response = s3_client.upload_file(
                S3_info['S3_key'], BUCKET_NAME, S3_info['S3_key'])
        else:
            print('이미 모델 있음')
    else:
        try:
            uvicorn.run("app:app", host='0.0.0.0', port=8002, reload=True)
        finally:
            requests.get('http://localhost:8080/flclient_out')

#
