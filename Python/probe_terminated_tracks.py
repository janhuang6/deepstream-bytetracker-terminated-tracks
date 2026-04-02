import gi

# Initialize GStreamer bindings
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
import pyds
import os
import ctypes
import numpy as np
import pickle

def create_element(element_type: str, element_name: str):
    """
    Helper function to create a GStreamer element.
    """
    element = Gst.ElementFactory.make(element_type, element_name)
    if not element:
        print(f"Failed to create {element_type} {element_name}")
    return element

def handle_bus_message(bus, message, loop):
    """
    Handles GStreamer bus messages (EOS, WARNING, ERROR, etc.).
    """
    msg_type = message.type
    if msg_type == Gst.MessageType.EOS:
        loop.quit()
        return True

    if msg_type == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        print(f'{err}: {debug}')
        return True

    if msg_type == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f'{err}: {debug}')
        loop.quit()
        return True

    return False

def probe_terminated_tracks(user_meta: pyds.NvDsUserMeta):
    """
    Probes and prints information about terminated tracks from user metadata.

    This function processes user metadata for terminated object lists. If the 
    metadata type matches NVDS_TRACKER_TERMINATED_LIST_META, it will cast the 
    user_meta_data to NvDsTargetMiscDataBatch and iterate through the batch 
    and stream objects, printing the class ID and unique object ID for each 
    terminated track.
    """
    if user_meta.base_meta.meta_type != pyds.NvDsMetaType.NVDS_TRACKER_TERMINATED_LIST_META:
        return False

    misc_data_batch = pyds.NvDsTargetMiscDataBatch.cast(user_meta.user_meta_data)
    for misc_data_stream in pyds.NvDsTargetMiscDataBatch.list(misc_data_batch):
        for misc_data_obj in pyds.NvDsTargetMiscDataStream.list(misc_data_stream):
            print(f"Python probed terminated track classId = {misc_data_obj.classId}, objectId = {misc_data_obj.uniqueId}")
    return

def probe_func(pad, info, u_data):
    """
    Pad probe callback function for post-processing batch metadata.
    This function is attached to the sink pad and processes frame and object metadata.
    It also iterates through user metadata to probe terminated tracks.
    """
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer")
        return Gst.PadProbeReturn.DROP

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    # Iterate through each frame in the batch
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        results = []
        # Iterate through each object in the frame
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            # Extract bounding box and object tracking ID
            x, y, w, h = (
                obj_meta.rect_params.left,
                obj_meta.rect_params.top,
                obj_meta.rect_params.width,
                obj_meta.rect_params.height,
            )
            bbox = (int(x), int(y), int(x + w), int(y + h))
            obj_id = obj_meta.object_id
            class_id = obj_meta.class_id

            # Check for person detection (class_id 0)
            if class_id == 0:
                print(f"Person detected: ID={obj_id}, bbox={bbox}")
                results.append({"type": "person", "object_id": obj_id, "bbox": bbox})

            # Check for cart detection (class_id 1)
            elif class_id == 1:
                print(f"Cart detected: ID={obj_id}, bbox={bbox}")
                results.append({"type": "cart", "object_id": obj_id, "bbox": bbox})

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    # --- Probe terminated tracks info ---
    # Iterate through user metadata in the batch and call probe_terminated_tracks
    l_user: Optional[pyds.GList] = batch_meta.batch_user_meta_list
    while l_user is not None:
        try:
            user_meta: pyds.NvDsUserMeta = pyds.NvDsUserMeta.cast(l_user.data)
        except StopIteration:
            break

        # This probes info from terminated tracks and prints out their details
        probe_terminated_tracks(user_meta)

        try:
            l_user = l_user.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK

class DeepStreamPipeline:
    """
    DeepStreamPipeline class sets up the GStreamer pipeline for video analytics.
    """
    def __init__(self):
        self.pipeline = None
        self.loop = None
        self.attach_ts = False
        self.is_live = 0
        self.initialize_pipeline()

    def initialize_pipeline(self):
        """
        Initializes the GStreamer pipeline, creates elements, links them,
        and sets properties.
        """
        Gst.init(None)
        self.pipeline = Gst.Pipeline()
        if not self.pipeline:
            print("Unable to create Pipeline")
            return

        # Create nvstreammux instance
        streammux = create_element("nvstreammux", "Stream-muxer")
        self.pipeline.add(streammux)

        # Add video source
        i = 0
        url = "file:///videos/video_8299_192.168.86.26_1739934000000.mp4"
        source_bin = self.create_source_bin(i, url)
        if not source_bin:
            print("Unable to create source bin")
        self.pipeline.add(source_bin)
        padname = f"sink_{i}"
        sinkpad = streammux.get_request_pad(padname)
        if not sinkpad:
            print("Unable to create sink pad bin")
        srcpad = source_bin.get_static_pad("src")
        if not srcpad:
            print("Unable to create src pad bin")
        srcpad.link(sinkpad)

        self.set_streammux_properties(streammux)

        # Add inference elements for human, cart and item detection
        pgie_human_detection = create_element("nvinfer", "human-detection")
        pgie_human_detection.set_property("config-file-path", "/configs/config_human_pose.txt")
        self.pipeline.add(pgie_human_detection)

        pgie_cart_detection = create_element("nvinfer", "cart-detection")
        pgie_cart_detection.set_property("config-file-path", "/configs/config_cart_detection.txt")
        pgie_cart_detection.set_property("config-file-path", "/configs/test_cart.txt")
        self.pipeline.add(pgie_cart_detection)

        pgie_item_detection = create_element("nvinfer", "item-detection")
        pgie_item_detection.set_property("config-file-path", "/configs/config_item_detection.txt")
        pgie_item_detection.set_property("config-file-path", "/configs/test_item.txt")
        self.pipeline.add(pgie_item_detection)

        # Create tracker element for object tracking
        tracker = create_element("nvtracker", "tracker")
        self.configure_tracker(tracker)
        self.pipeline.add(tracker)

        # Add sink (output)
        sink = create_element("fakesink", "fakesink")
        sink.set_property("sync", False)
        self.pipeline.add(sink)

        # Link pipeline elements in order: streammux -> inference -> tracker -> sink
        streammux.link(pgie_human_detection)
        pgie_human_detection.link(pgie_cart_detection)
        pgie_cart_detection.link(pgie_item_detection)
        pgie_item_detection.link(tracker)
        tracker.link(sink)

        # Attach pad probe to the sink for post-processing (including terminated tracks info)
        sink_pad = sink.get_static_pad("sink")
        sink_pad.add_probe(Gst.PadProbeType.BUFFER, probe_func, 0)

    def create_source_bin(self, self_, index, uri):
        """
        Creates a GStreamer source bin for input video URI.
        """
        print(f"Creating source bin {index} {uri}")
        bin_name = f"source-bin-{index}"
        nbin = Gst.Bin.new(bin_name)
        if not nbin:
            print("Unable to create source bin")
            return None

        uri_decode_bin = create_element("uridecodebin", f"uri-decode-bin-{index}")
        uri_decode_bin.set_property("uri", uri)
        uri_decode_bin.connect("pad-added", self.handle_new_pad, nbin)

        Gst.Bin.add(nbin, uri_decode_bin)
        bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
        if not bin_pad:
            print("Failed to add ghost pad in source bin")
            return None
        return nbin

    def configure_tracker(self, tracker):
        """
        Sets DeepStream tracker properties.
        """
        config = {
            'tracker-width': 640,
            'tracker-height': 384,
            'gpu-id': 0,
            'll-lib-file': '/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmmultiobjecttracker.so',
            'll-lib-file': '/opt/nvidia/deepstream/deepstream/lib/libnvdsgst_bytracker.so',
            'll-config-file': '/configs/config_tracker.yml',
        }
        for key, val in config.items():
            tracker.set_property(key, val)

    def handle_new_pad(self, self_, decode_bin, pad, bin):
        """
        Handles newly added pads for decodebin (video source).
        Ensures only NVIDIA decoder is linked to source bin.
        """
        caps = pad.get_current_caps() or pad.query_caps()
        struct = caps.get_structure(0)
        name = struct.get_name()
        features = caps.get_features(0)

        if "video" in name:
            if features.contains("memory:NVMM"):
                ghost_pad = bin.get_static_pad("src")
                if not ghost_pad.set_target(pad):
                    print("Failed to link decoder src pad to source bin ghost pad")
            else:
                print("Decodebin did not pick nvidia decoder plugin.")

    def set_streammux_properties(self, streammux):
        """
        Sets the properties for the streammux element.
        """
        streammux.set_property('width', 1920)
        streammux.set_property('height', 1080)
        streammux.set_property('live-source', self.is_live)
        streammux.set_property('batch-size', 1)
        streammux.set_property('attach-sys-ts', self.attach_ts)
        streammux.set_property('sync_inputs', True)
        streammux.set_property('batched-push-timeout', 33333)

    def main_loop(self):
        """
        Starts the GStreamer main event loop and the pipeline.
        """
        self.loop = GLib.MainLoop()
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", handle_bus_message, self.loop)

        self.pipeline.set_state(Gst.State.PLAYING)
        print("Start Deepstream Pipeline")
        self.loop.run()

    def stop(self):
        """
        Stops the pipeline and sets it to NULL state.
        """
        self.pipeline.set_state(Gst.State.NULL)

if __name__ == "__main__":
    pipe = DeepStreamPipeline()
    pipe.main_loop()