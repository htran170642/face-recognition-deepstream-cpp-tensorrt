# Deepstream app with Retinace and Arcface

## 1. Prerequisites:
Follow these procedures to use the deepstream-app application for native
compilation.

You must have the following development packages installed

    GStreamer-1.0
    GStreamer-1.0 Base Plugins
    GStreamer-1.0 gstrtspserver
    X11 client-side library
    Glib json library - json-glib-1.0
    yaml-cpp

To install these packages, execute the following command:
   sudo apt-get install libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev \
   libgstrtspserver-1.0-dev libx11-dev libjson-glib-dev libyaml-cpp-dev

## 2. To compile and run deepstream app
```bash
$ mkdir build && cd build && cmake ..
$ make -j12
```

## 3. Usage:
  Run the application by executing the command:
   ./src/deepstream-face-recognition -c ../config.json

Please refer `config.json` to modify your local config path

## 4 Benchmark
   - send a crop face from gst-dsfacesearch plugin <br/>
   (2023-01-12 04:05:08) [INFO    ] Request: 127.0.0.1:46394 0x7f4d04002220 HTTP/1.1 POST /recognize <br/>
   (2023-01-12 04:05:08) [INFO    ] Image: [112 x 112] <br/>
   (2023-01-12 04:05:08) [INFO    ] Getting embedding... <br/>
   (2023-01-12 04:05:08) [INFO    ] Feature matching... <br/>
   (2023-01-12 04:05:08) [INFO    ] Prediction: hiep 0.894664 <br/>
   (2023-01-12 04:05:08) [INFO    ] time: 3ms <br/>
   (2023-01-12 04:05:08) [INFO    ] Response: 0x7f4d04002220 /recognize 200 0 <br/>

Note:
   - `gst-dsfacesearch` plugins will call api for face recognition (implemented in `tensorrt` folder)
   - Disable [ds-facesearch] and enable [secondary-gie0] if you want to run both detection and recognition in Deepstream app
