import numpy as np
try:
    import pyrealsense2 as rs
    pipeline = rs.pipeline()
    config = rs.config()
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    pipeline.start(config)
except:
    print('L515 not found')

def RGB_L515():
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
    return color_image