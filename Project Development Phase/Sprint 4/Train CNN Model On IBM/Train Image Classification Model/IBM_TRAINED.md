```python
!pip install keras==2.2.4
!pip install tensorflow==2.7.3
```

    Collecting keras==2.2.4
      Downloading Keras-2.2.4-py2.py3-none-any.whl (312 kB)
    [K     |████████████████████████████████| 312 kB 21.0 MB/s eta 0:00:01
    [?25hRequirement already satisfied: scipy>=0.14 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from keras==2.2.4) (1.7.3)
    Requirement already satisfied: h5py in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from keras==2.2.4) (3.2.1)
    Requirement already satisfied: six>=1.9.0 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from keras==2.2.4) (1.15.0)
    Requirement already satisfied: keras-preprocessing>=1.0.5 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from keras==2.2.4) (1.1.2)
    Requirement already satisfied: numpy>=1.9.1 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from keras==2.2.4) (1.20.3)
    Requirement already satisfied: pyyaml in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from keras==2.2.4) (5.4.1)
    Collecting keras-applications>=1.0.6
      Downloading Keras_Applications-1.0.8-py3-none-any.whl (50 kB)
    [K     |████████████████████████████████| 50 kB 16.6 MB/s eta 0:00:01
    [?25hInstalling collected packages: keras-applications, keras
      Attempting uninstall: keras
        Found existing installation: keras 2.7.0
        Uninstalling keras-2.7.0:
          Successfully uninstalled keras-2.7.0
    [31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    tensorflow 2.7.2 requires keras<2.8,>=2.7.0, but you have keras 2.2.4 which is incompatible.[0m
    Successfully installed keras-2.2.4 keras-applications-1.0.8
    Collecting tensorflow==2.7.3
      Downloading tensorflow-2.7.3-cp39-cp39-manylinux2010_x86_64.whl (495.6 MB)
    [K     |████████████████████████████████| 495.6 MB 53 kB/s s eta 0:00:01|████▉                           | 74.1 MB 14.6 MB/s eta 0:00:29�█████████▌                 | 225.4 MB 75.0 MB/s eta 0:00:04MB/s eta 0:00:02MB/s eta 0:00:02��██▌     | 410.2 MB 84.7 MB/s eta 0:00:02
    [?25hRequirement already satisfied: tensorflow-estimator<2.8,~=2.7.0rc0 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from tensorflow==2.7.3) (2.7.0)
    Requirement already satisfied: termcolor>=1.1.0 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from tensorflow==2.7.3) (1.1.0)
    Requirement already satisfied: wheel<1.0,>=0.32.0 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from tensorflow==2.7.3) (0.37.0)
    Requirement already satisfied: tensorboard~=2.6 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from tensorflow==2.7.3) (2.7.0)
    Requirement already satisfied: wrapt>=1.11.0 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from tensorflow==2.7.3) (1.12.1)
    Requirement already satisfied: opt-einsum>=2.3.2 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from tensorflow==2.7.3) (3.3.0)
    Requirement already satisfied: astunparse>=1.6.0 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from tensorflow==2.7.3) (1.6.3)
    Requirement already satisfied: typing-extensions>=3.6.6 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from tensorflow==2.7.3) (4.1.1)
    Requirement already satisfied: six>=1.12.0 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from tensorflow==2.7.3) (1.15.0)
    Requirement already satisfied: protobuf<3.20,>=3.9.2 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from tensorflow==2.7.3) (3.19.1)
    Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.21.0 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from tensorflow==2.7.3) (0.23.1)
    Requirement already satisfied: gast<0.5.0,>=0.2.1 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from tensorflow==2.7.3) (0.4.0)
    Requirement already satisfied: numpy>=1.14.5 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from tensorflow==2.7.3) (1.20.3)
    Requirement already satisfied: grpcio<2.0,>=1.24.3 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from tensorflow==2.7.3) (1.42.0)
    Requirement already satisfied: absl-py>=0.4.0 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from tensorflow==2.7.3) (0.12.0)
    Requirement already satisfied: keras-preprocessing>=1.1.1 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from tensorflow==2.7.3) (1.1.2)
    Collecting libclang>=9.0.1
      Downloading libclang-14.0.6-py2.py3-none-manylinux2010_x86_64.whl (14.1 MB)
    [K     |████████████████████████████████| 14.1 MB 73.7 MB/s eta 0:00:01
    [?25hRequirement already satisfied: google-pasta>=0.1.1 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from tensorflow==2.7.3) (0.2.0)
    Collecting keras<2.8,>=2.7.0rc0
      Downloading keras-2.7.0-py2.py3-none-any.whl (1.3 MB)
    [K     |████████████████████████████████| 1.3 MB 82.1 MB/s eta 0:00:01
    [?25hRequirement already satisfied: h5py>=2.9.0 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from tensorflow==2.7.3) (3.2.1)
    Requirement already satisfied: flatbuffers<3.0,>=1.12 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from tensorflow==2.7.3) (2.0)
    Requirement already satisfied: google-auth<3,>=1.6.3 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from tensorboard~=2.6->tensorflow==2.7.3) (1.23.0)
    Requirement already satisfied: requests<3,>=2.21.0 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from tensorboard~=2.6->tensorflow==2.7.3) (2.26.0)
    Requirement already satisfied: tensorboard-data-server<0.7.0,>=0.6.0 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from tensorboard~=2.6->tensorflow==2.7.3) (0.6.1)
    Requirement already satisfied: setuptools>=41.0.0 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from tensorboard~=2.6->tensorflow==2.7.3) (58.0.4)
    Requirement already satisfied: markdown>=2.6.8 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from tensorboard~=2.6->tensorflow==2.7.3) (3.3.3)
    Requirement already satisfied: google-auth-oauthlib<0.5,>=0.4.1 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from tensorboard~=2.6->tensorflow==2.7.3) (0.4.4)
    Requirement already satisfied: tensorboard-plugin-wit>=1.6.0 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from tensorboard~=2.6->tensorflow==2.7.3) (1.6.0)
    Requirement already satisfied: werkzeug>=0.11.15 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from tensorboard~=2.6->tensorflow==2.7.3) (2.0.2)
    Requirement already satisfied: rsa<5,>=3.1.4 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from google-auth<3,>=1.6.3->tensorboard~=2.6->tensorflow==2.7.3) (4.7.2)
    Requirement already satisfied: cachetools<5.0,>=2.0.0 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from google-auth<3,>=1.6.3->tensorboard~=2.6->tensorflow==2.7.3) (4.2.2)
    Requirement already satisfied: pyasn1-modules>=0.2.1 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from google-auth<3,>=1.6.3->tensorboard~=2.6->tensorflow==2.7.3) (0.2.8)
    Requirement already satisfied: requests-oauthlib>=0.7.0 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard~=2.6->tensorflow==2.7.3) (1.3.0)
    Requirement already satisfied: pyasn1<0.5.0,>=0.4.6 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard~=2.6->tensorflow==2.7.3) (0.4.8)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from requests<3,>=2.21.0->tensorboard~=2.6->tensorflow==2.7.3) (2022.9.24)
    Requirement already satisfied: charset-normalizer~=2.0.0 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from requests<3,>=2.21.0->tensorboard~=2.6->tensorflow==2.7.3) (2.0.4)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from requests<3,>=2.21.0->tensorboard~=2.6->tensorflow==2.7.3) (1.26.7)
    Requirement already satisfied: idna<4,>=2.5 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from requests<3,>=2.21.0->tensorboard~=2.6->tensorflow==2.7.3) (3.3)
    Requirement already satisfied: oauthlib>=3.0.0 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard~=2.6->tensorflow==2.7.3) (3.2.1)
    Installing collected packages: libclang, keras, tensorflow
      Attempting uninstall: keras
        Found existing installation: Keras 2.2.4
        Uninstalling Keras-2.2.4:
          Successfully uninstalled Keras-2.2.4
      Attempting uninstall: tensorflow
        Found existing installation: tensorflow 2.7.2
        Uninstalling tensorflow-2.7.2:
          Successfully uninstalled tensorflow-2.7.2
    Successfully installed keras-2.7.0 libclang-14.0.6 tensorflow-2.7.3



```python
pwd
```




    '/home/wsuser/work'




```python

import os, types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share the notebook.
cos_client = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='f6frDnV1lV-nAfp8RCfFmF3FC_ZoOTCv6ihpYGZeOpfX',
    ibm_auth_endpoint="https://iam.cloud.ibm.com/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3.private.us.cloud-object-storage.appdomain.cloud')

bucket = 'predictions-donotdelete-pr-s1jm5hqaxalspn'
object_key = 'conversation engine for deaf and dumb .zip'

streaming_body_2 = cos_client.get_object(Bucket=bucket, Key=object_key)['Body']

# Your data file was loaded into a botocore.response.StreamingBody object.
# Please read the documentation of ibm_boto3 and pandas to learn more about the possibilities to load the data.
# ibm_boto3 documentation: https://ibm.github.io/ibm-cos-sdk-python/
# pandas documentation: http://pandas.pydata.org/

```


```python
from io import BytesIO
import zipfile
unzip = zipfile.ZipFile(BytesIO(streaming_body_2.read()),'r')
file_paths = unzip.namelist()
for path in file_paths:
    unzip.extract(path)
```


```python
import os
filenames = os.listdir('/home/wsuser/work/conversation engine for deaf and dumb')
```

# **Image Preprocessing**

### Import ImageDataGenerator Library And Configure It


```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```


```python
# Training Datagen
train_datagen = ImageDataGenerator(rescale=1/255,zoom_range=0.2,horizontal_flip=True,vertical_flip=False)
# Testing Datagen
test_datagen = ImageDataGenerator(rescale=1/255)

```


```python
import tensorflow as tf
import os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as display
from PIL import Image
import pathlib
```


```python
# Training Dataset
x_train=train_datagen.flow_from_directory('/home/wsuser/work/conversation engine for deaf and dumb/Dataset/training_set',target_size=(64,64), class_mode='categorical',batch_size=900,color_mode = "grayscale")
# Testing Dataset
x_test=test_datagen.flow_from_directory('/home/wsuser/work/conversation engine for deaf and dumb/Dataset/test_set',target_size=(64,64), class_mode='categorical',batch_size=900,color_mode = "grayscale")

```

    Found 15750 images belonging to 9 classes.
    Found 2250 images belonging to 9 classes.


##Apply ImageDataGenerator Functionality To Train And Test Set


```python
print("Len x-train : ",len(x_train))
print("Len x-test : ", len(x_test))
```

    Len x-train :  18
    Len x-test :  3



```python
# The Class Indices in Training Dataset
x_train.class_indices
```




    {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8}




```python
# The Class Indices in Test Dataset
x_test.class_indices
```




    {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8}



#**Model building**

##Import The Required Model Building Libraries


```python
# Importing Libraries
from keras.models import Sequential
from keras.layers import Convolution2D,MaxPooling2D,Flatten,Dense
```

##Initialize The Model


```python
# Creating Model
model=Sequential()
```

##Add The Convolution Layer


```python
# Adding The Convolution Layer
model.add(Convolution2D(32,(3,3),input_shape=(64,64,1),activation='relu'))
```

##Add The Pooling Layer


```python
# Adding The Pooling Layer
model.add(MaxPooling2D(pool_size=(2,2)))
```

##Add The Flatten Layer


```python
# Adding The Flatten Layer
model.add(Flatten())
```

## Add the Dense Layers


```python
# Adding Dense Layers
model.add(Dense(512,activation='relu'))
model.add(Dense(9,activation='softmax'))
```

## Compilie the Model


```python
# Compiling the Model
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
```


```python
# Fitting the Model Generator
model.fit_generator(x_train,steps_per_epoch=len(x_train),epochs=10,validation_data=x_test,validation_steps=len(x_test))
```

    /tmp/wsuser/ipykernel_164/1042518445.py:2: UserWarning: `Model.fit_generator` is deprecated and will be removed in a future version. Please use `Model.fit`, which supports generators.
      model.fit_generator(x_train,steps_per_epoch=len(x_train),epochs=10,validation_data=x_test,validation_steps=len(x_test))


    Epoch 1/10
    18/18 [==============================] - 40s 2s/step - loss: 0.0137 - accuracy: 0.9978 - val_loss: 0.2479 - val_accuracy: 0.9778
    Epoch 2/10
    18/18 [==============================] - 40s 2s/step - loss: 0.0117 - accuracy: 0.9980 - val_loss: 0.2293 - val_accuracy: 0.9769
    Epoch 3/10
    18/18 [==============================] - 39s 2s/step - loss: 0.0093 - accuracy: 0.9988 - val_loss: 0.2329 - val_accuracy: 0.9769
    Epoch 4/10
    18/18 [==============================] - 39s 2s/step - loss: 0.0072 - accuracy: 0.9997 - val_loss: 0.2472 - val_accuracy: 0.9782
    Epoch 5/10
    18/18 [==============================] - 39s 2s/step - loss: 0.0060 - accuracy: 0.9993 - val_loss: 0.2520 - val_accuracy: 0.9773
    Epoch 6/10
    18/18 [==============================] - 39s 2s/step - loss: 0.0062 - accuracy: 0.9992 - val_loss: 0.2563 - val_accuracy: 0.9773
    Epoch 7/10
    18/18 [==============================] - 39s 2s/step - loss: 0.0051 - accuracy: 0.9994 - val_loss: 0.2713 - val_accuracy: 0.9778
    Epoch 8/10
    18/18 [==============================] - 39s 2s/step - loss: 0.0044 - accuracy: 0.9993 - val_loss: 0.2697 - val_accuracy: 0.9773
    Epoch 9/10
    18/18 [==============================] - 39s 2s/step - loss: 0.0038 - accuracy: 0.9996 - val_loss: 0.2866 - val_accuracy: 0.9773
    Epoch 10/10
    18/18 [==============================] - 39s 2s/step - loss: 0.0033 - accuracy: 0.9996 - val_loss: 0.2710 - val_accuracy: 0.9769





    <keras.callbacks.History at 0x7fe01cef6790>



##Fit And Save The Model


```python
pwd

```




    '/home/wsuser/work'




```python
# Saving The Model
model.save('trained_on_ibm.h5')
```


```python
!tar zcvf IBM-model.tgz trained_on_ibm.h5
```

    trained_on_ibm.h5



```python
ls -1
```

    [0m[01;34m'conversation engine for deaf and dumb'[0m/
    IBM-model.tgz
    [01;34mibm_trained[0m/
    [01;34mIBM_trained[0m/
    [01;34mibm-trained-model[0m/
    Sign-language-model.tgz
    trained_on_ibm.h5



```python
!pip install watson-machine-learning-client --upgrade
```

    Collecting watson-machine-learning-client
      Downloading watson_machine_learning_client-1.0.391-py3-none-any.whl (538 kB)
    [K     |████████████████████████████████| 538 kB 23.4 MB/s eta 0:00:01
    [?25hRequirement already satisfied: tqdm in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from watson-machine-learning-client) (4.62.3)
    Requirement already satisfied: ibm-cos-sdk in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from watson-machine-learning-client) (2.11.0)
    Requirement already satisfied: requests in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from watson-machine-learning-client) (2.26.0)
    Requirement already satisfied: lomond in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from watson-machine-learning-client) (0.3.3)
    Requirement already satisfied: certifi in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from watson-machine-learning-client) (2022.9.24)
    Requirement already satisfied: tabulate in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from watson-machine-learning-client) (0.8.9)
    Requirement already satisfied: boto3 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from watson-machine-learning-client) (1.18.21)
    Requirement already satisfied: pandas in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from watson-machine-learning-client) (1.3.4)
    Requirement already satisfied: urllib3 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from watson-machine-learning-client) (1.26.7)
    Requirement already satisfied: botocore<1.22.0,>=1.21.21 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from boto3->watson-machine-learning-client) (1.21.41)
    Requirement already satisfied: jmespath<1.0.0,>=0.7.1 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from boto3->watson-machine-learning-client) (0.10.0)
    Requirement already satisfied: s3transfer<0.6.0,>=0.5.0 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from boto3->watson-machine-learning-client) (0.5.0)
    Requirement already satisfied: python-dateutil<3.0.0,>=2.1 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from botocore<1.22.0,>=1.21.21->boto3->watson-machine-learning-client) (2.8.2)
    Requirement already satisfied: six>=1.5 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from python-dateutil<3.0.0,>=2.1->botocore<1.22.0,>=1.21.21->boto3->watson-machine-learning-client) (1.15.0)
    Requirement already satisfied: ibm-cos-sdk-core==2.11.0 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from ibm-cos-sdk->watson-machine-learning-client) (2.11.0)
    Requirement already satisfied: ibm-cos-sdk-s3transfer==2.11.0 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from ibm-cos-sdk->watson-machine-learning-client) (2.11.0)
    Requirement already satisfied: charset-normalizer~=2.0.0 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from requests->watson-machine-learning-client) (2.0.4)
    Requirement already satisfied: idna<4,>=2.5 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from requests->watson-machine-learning-client) (3.3)
    Requirement already satisfied: pytz>=2017.3 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from pandas->watson-machine-learning-client) (2021.3)
    Requirement already satisfied: numpy>=1.17.3 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from pandas->watson-machine-learning-client) (1.20.3)
    Installing collected packages: watson-machine-learning-client
    Successfully installed watson-machine-learning-client-1.0.391



```python
from ibm_watson_machine_learning import APIClient
wml_credentials ={
    "url":"https://us-south.ml.cloud.ibm.com",
    "apikey":"65yy2Zf2RrLpAmFe2T29AAfcRuaKhTh4wLWfpuAHsP32"
}
client = APIClient(wml_credentials)
```


```python
client = APIClient(wml_credentials)
```


```python
def guid_from_space_name(client, space_name):
    space = client.spaces.get_details()
    return(next(item for item in space['resources'] if item['entity']['name'] == space_name)['metadata']['id'])
```


```python
space_uid = guid_from_space_name(client, 'IBM project final model')
print("Space UID = " + space_uid)
```

    Space UID = a53c7e6c-2504-4e1f-91d1-42d108bf9e05



```python
client.set.default_space(space_uid)
```




    'SUCCESS'




```python
client.software_specifications.list()
```

    -----------------------------  ------------------------------------  ----
    NAME                           ASSET_ID                              TYPE
    default_py3.6                  0062b8c9-8b7d-44a0-a9b9-46c416adcbd9  base
    kernel-spark3.2-scala2.12      020d69ce-7ac1-5e68-ac1a-31189867356a  base
    pytorch-onnx_1.3-py3.7-edt     069ea134-3346-5748-b513-49120e15d288  base
    scikit-learn_0.20-py3.6        09c5a1d0-9c1e-4473-a344-eb7b665ff687  base
    spark-mllib_3.0-scala_2.12     09f4cff0-90a7-5899-b9ed-1ef348aebdee  base
    pytorch-onnx_rt22.1-py3.9      0b848dd4-e681-5599-be41-b5f6fccc6471  base
    ai-function_0.1-py3.6          0cdb0f1e-5376-4f4d-92dd-da3b69aa9bda  base
    shiny-r3.6                     0e6e79df-875e-4f24-8ae9-62dcc2148306  base
    tensorflow_2.4-py3.7-horovod   1092590a-307d-563d-9b62-4eb7d64b3f22  base
    pytorch_1.1-py3.6              10ac12d6-6b30-4ccd-8392-3e922c096a92  base
    tensorflow_1.15-py3.6-ddl      111e41b3-de2d-5422-a4d6-bf776828c4b7  base
    autoai-kb_rt22.2-py3.10        125b6d9a-5b1f-5e8d-972a-b251688ccf40  base
    runtime-22.1-py3.9             12b83a17-24d8-5082-900f-0ab31fbfd3cb  base
    scikit-learn_0.22-py3.6        154010fa-5b3b-4ac1-82af-4d5ee5abbc85  base
    default_r3.6                   1b70aec3-ab34-4b87-8aa0-a4a3c8296a36  base
    pytorch-onnx_1.3-py3.6         1bc6029a-cc97-56da-b8e0-39c3880dbbe7  base
    kernel-spark3.3-r3.6           1c9e5454-f216-59dd-a20e-474a5cdf5988  base
    pytorch-onnx_rt22.1-py3.9-edt  1d362186-7ad5-5b59-8b6c-9d0880bde37f  base
    tensorflow_2.1-py3.6           1eb25b84-d6ed-5dde-b6a5-3fbdf1665666  base
    spark-mllib_3.2                20047f72-0a98-58c7-9ff5-a77b012eb8f5  base
    tensorflow_2.4-py3.8-horovod   217c16f6-178f-56bf-824a-b19f20564c49  base
    runtime-22.1-py3.9-cuda        26215f05-08c3-5a41-a1b0-da66306ce658  base
    do_py3.8                       295addb5-9ef9-547e-9bf4-92ae3563e720  base
    autoai-ts_3.8-py3.8            2aa0c932-798f-5ae9-abd6-15e0c2402fb5  base
    tensorflow_1.15-py3.6          2b73a275-7cbf-420b-a912-eae7f436e0bc  base
    kernel-spark3.3-py3.9          2b7961e2-e3b1-5a8c-a491-482c8368839a  base
    pytorch_1.2-py3.6              2c8ef57d-2687-4b7d-acce-01f94976dac1  base
    spark-mllib_2.3                2e51f700-bca0-4b0d-88dc-5c6791338875  base
    pytorch-onnx_1.1-py3.6-edt     32983cea-3f32-4400-8965-dde874a8d67e  base
    spark-mllib_3.0-py37           36507ebe-8770-55ba-ab2a-eafe787600e9  base
    spark-mllib_2.4                390d21f8-e58b-4fac-9c55-d7ceda621326  base
    autoai-ts_rt22.2-py3.10        396b2e83-0953-5b86-9a55-7ce1628a406f  base
    xgboost_0.82-py3.6             39e31acd-5f30-41dc-ae44-60233c80306e  base
    pytorch-onnx_1.2-py3.6-edt     40589d0e-7019-4e28-8daa-fb03b6f4fe12  base
    pytorch-onnx_rt22.2-py3.10     40e73f55-783a-5535-b3fa-0c8b94291431  base
    default_r36py38                41c247d3-45f8-5a71-b065-8580229facf0  base
    autoai-ts_rt22.1-py3.9         4269d26e-07ba-5d40-8f66-2d495b0c71f7  base
    autoai-obm_3.0                 42b92e18-d9ab-567f-988a-4240ba1ed5f7  base
    pmml-3.0_4.3                   493bcb95-16f1-5bc5-bee8-81b8af80e9c7  base
    spark-mllib_2.4-r_3.6          49403dff-92e9-4c87-a3d7-a42d0021c095  base
    xgboost_0.90-py3.6             4ff8d6c2-1343-4c18-85e1-689c965304d3  base
    pytorch-onnx_1.1-py3.6         50f95b2a-bc16-43bb-bc94-b0bed208c60b  base
    autoai-ts_3.9-py3.8            52c57136-80fa-572e-8728-a5e7cbb42cde  base
    spark-mllib_2.4-scala_2.11     55a70f99-7320-4be5-9fb9-9edb5a443af5  base
    spark-mllib_3.0                5c1b0ca2-4977-5c2e-9439-ffd44ea8ffe9  base
    autoai-obm_2.0                 5c2e37fa-80b8-5e77-840f-d912469614ee  base
    spss-modeler_18.1              5c3cad7e-507f-4b2a-a9a3-ab53a21dee8b  base
    cuda-py3.8                     5d3232bf-c86b-5df4-a2cd-7bb870a1cd4e  base
    runtime-22.2-py3.10-xc         5e8cddff-db4a-5a6a-b8aa-2d4af9864dab  base
    autoai-kb_3.1-py3.7            632d4b22-10aa-5180-88f0-f52dfb6444d7  base
    -----------------------------  ------------------------------------  ----
    Note: Only first 50 records were displayed. To display more use 'limit' parameter.



```python
software_spec_uid = client.software_specifications.get_uid_by_name("tensorflow_rt22.1-py3.9")
software_spec_uid
```




    'acd9c798-6974-5d2f-a657-ce06e986df4d'




```python
model_details = client.repository.store_model(model='IBM-model.tgz', meta_props={
    client.repository.ModelMetaNames.NAME: "CNN",
    client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: software_spec_uid,
    client.repository.ModelMetaNames.TYPE: "tensorflow_2.7"})
model_id = client.repository.get_model_id(model_details)
```


```python
model_id
```




    '3317859f-71e7-4edd-84d6-7f8c86456745'



#Test The Model


##Import The Packages And Load The Saved Model


```python
import numpy as np
from keras.models import load_model
import cv2
```

##Load The Test Image, Pre-Process It And Predict



```python
pwd

```




    '/home/wsuser/work'




```python
model=load_model('trained_on_ibm.h5')
```


```python
from skimage.transform import resize
def detect(frame):
  img = resize(frame, (64,64,1))
  img = np.expand_dims(img,axis=0)
  if(np.max(img)>1):
    img = img/255.0
  prediction = model.predict(img)
  predictions = np.argmax(model.predict(img),axis=1)
  print(prediction)
  print(predictions)
  predicted = list(predictions)
  index=['A','B','C','D','E','F','G','H','I']
  print(index[predicted[0]-1])

```


```python
frame = cv2.imread('/home/wsuser/work/conversation engine for deaf and dumb/Dataset/test_set/D/107.png')
data = detect(frame)
```

    [[0.08970424 0.09270186 0.10576255 0.10954611 0.13942339 0.11134484
      0.10993771 0.12809943 0.11347996]]
    [4]
    D

