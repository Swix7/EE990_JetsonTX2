import logging
import os
import yaml
import numpy as np

from lib.mpfps import FPS


def load_config():
    """
    LOAD CONFIG FILE
    Convert config.yml to DICT.
    """
    cfg = None
    if (os.path.isfile('config.yml')):
        with open("config.yml", 'r') as ymlfile:
            cfg = yaml.safe_load(ymlfile)
    else:
        raise FileNotFoundError(("File not found: config.yml"))
    cfg.update({'src_from': 'movie'})
    return cfg

def log_format(debug_mode):
    """
    LOG FORMAT
    If debug_mode, show detailed log
    """
    if debug_mode:
        np.set_printoptions(precision=5, suppress=True, threshold=np.inf)  # suppress scientific float notation
        logging.basicConfig(level=logging.DEBUG,
                            format='[%(levelname)s] time:%(created).8f pid:%(process)d pn:%(processName)-10s tid:%(thread)d tn:%(threadName)-10s fn:%(funcName)-10s %(message)s',
        )
    return


def main():
    try:
        """
        LOAD SETUP VARIABLES
        """
        cfg = load_config()
        debug_mode = cfg['debug_mode']
      

        """
        LOG FORMAT MODE
        """
        log_format(debug_mode)
#        download_model()

        """
        START DETECTION, FPS, FPS PRINT
        """
        fps = FPS(cfg)
        fps_counter_proc = fps.start_counter()
        fps_console_proc = fps.start_console()

        model = cfg['model']
        if model == 'mask_rcnn_resnet50_atrous_coco':
            from lib.detection_trt_mask_v1 import TRTV1
            detection = TRTV1()
            detection.start(cfg)
        elif model == 'faster_rcnn_resnet50_coco':
            from lib.detection_trt_v1 import TRTV1
            detection = TRTV1()
            detection.start(cfg)
        else:
        	raise IOError(("Unknown model."))

        fps_counter_proc.join()
        fps_console_proc.join()
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        pass

if __name__ == '__main__':
    main()