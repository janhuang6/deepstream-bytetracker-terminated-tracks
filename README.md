# deepstream-bytetracker-terminated-tracks
This project integrates a state-of-the-art ByteTracker into the NVIDIA DeepStream pipeline, modifying both C++ and Python layers to expose terminated track lists.

# Byte-Track Integration with Deepstream 6.4

**MODIFIED:** Integration and instructions updated for Deepstream 7.1 & 6.4, multi-stream and terminated tracks support.

This code is based on the [official code](https://github.com/ifzhang/ByteTrack).

Integrating Byte-Track C++ code with the Deepstream-6.4.

Features:
* Only support for DeepStream 6.4. For other verions, please refer to the [official code](https://github.com/ifzhang/ByteTrack).
* Support multi-streams
* The [memory leak issue](https://github.com/ifzhang/ByteTrack/issues/276) should be resolved.


https://github.com/Ending2015a/ByteTrack-DS6.4/assets/18180004/8e98ebf0-e9dd-4c9a-b845-810590b3f279


## Build Instructions
```
$mkdir build && cd build

$cmake ..

$make ByteTracker
```

This will create ./lib/libByteTracker.so file which can be passed as the custom low level tracker library to deepstream.
To do so just add it to the folder 
```
/opt/nvidia/deepstream/deepstream/lib/
```

In your deepstream_app_config.txt add the tracker.
```
[tracker]
enable=1
tracker-width=640
tracker-height=384
gpu-id=0
ll-lib-file=//opt/nvidia/deepstream/deepstream/lib/libByteTracker.so
enable-batch-process=1
```


## Modified/Enhanced Features

 **Multi-stream support:** Each stream gets its own ByteTracker instance.
 **Terminated tracks output:** See comments in code and the Python probe file.
 **DeepStream 7.1 support:** Update config, see `/isee/deepstream/configs/config_tracker.yml`


## Trying out Multiple Detectors
Feel free to try multiple detectors. You can do this by using https://github.com/marcoslucianops/DeepStream-Yolo

Now adding the tracker lines in the config file you should get your tracker working. 


## References
1. [How to Implement a Custom Low-Level Tracker Library in Deepstream](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html#how-to-implement-a-custom-low-level[...])
2. [Byte-Track](https://github.com/ifzhang/ByteTrack)
