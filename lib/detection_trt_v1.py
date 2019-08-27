import numpy as np
from lib.session_worker import SessionWorker
from lib.load_graph_trt_v1 import LoadFrozenGraph
from lib.load_label_map import LoadLabelMap
from lib.mpvariable import MPVariable
from lib.mpvisualizeworker import MPVisualizeWorker, visualization
from lib.mpio import start_sender
from lib.power import PowerRead, Device

import time
import cv2
import tensorflow as tf
import os

#TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

with open("Processing_Times.txt", "w+") as f:#Generates Headers
    f.write("Worker(s) Visualization(s) Total(s)\n")
    f.close()

import sys
PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3
if PY2:
    import Queue
elif PY3:
    import queue as Queue


class TRTV1():
    def __init__(self):
        return

    def start(self, cfg):
        """ """ """ """ """ """ """ """ """ """ """
        GET CONFIG
        """ """ """ """ """ """ """ """ """ """ """
        FORCE_GPU_COMPATIBLE = cfg['force_gpu_compatible']
        SAVE_TO_FILE         = cfg['save_to_file']
        VISUALIZE            = cfg['visualize']
        VIS_WORKER           = cfg['vis_worker']
        VIS_TEXT             = cfg['vis_text']
        MAX_FRAMES           = cfg['max_frames']
        WIDTH                = cfg['width']
        HEIGHT               = cfg['height']
        FPS_INTERVAL         = cfg['fps_interval']
        DET_INTERVAL         = cfg['det_interval']
        DET_TH               = cfg['det_th']
        LOG_DEVICE           = cfg['log_device']
        ALLOW_MEMORY_GROWTH  = cfg['allow_memory_growth']
        SPLIT_SHAPE          = cfg['split_shape']
        DEBUG_MODE           = cfg['debug_mode']
        LABEL_PATH           = cfg['label_path']
        NUM_CLASSES          = cfg['num_classes']
        SRC_FROM             = cfg['src_from']
        DEVICE               = eval(cfg['device'])
    
        MOVIE  = 1

        SRC_FROM = MOVIE
        VIDEO_INPUT = cfg['movie_input']
        """ """

        """ """ """ """ """ """ """ """ """ """ """
        LOAD FROZEN_GRAPH
        """ """ """ """ """ """ """ """ """ """ """
        load_frozen_graph = LoadFrozenGraph(cfg)
        graph = load_frozen_graph.load_graph()
        """ """

        """ """ """ """ """ """ """ """ """ """ """
        LOAD LABEL MAP
        """ """ """ """ """ """ """ """ """ """ """
        llm = LoadLabelMap()
        category_index = llm.load_label_map(cfg)
        """ """

        """ """ """ """ """ """ """ """ """ """ """
        PREPARE TF CONFIG OPTION
        """ """ """ """ """ """ """ """ """ """ """
        # Session Config: allow seperate GPU/CPU adressing and limit memory allocation
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        config.gpu_options.allow_growth = ALLOW_MEMORY_GROWTH
        config.gpu_options.force_gpu_compatible = FORCE_GPU_COMPATIBLE
        #config.gpu_options.visible_device_list = '0,1,2,3' # For GPU SERVER
        #config.gpu_options.per_process_gpu_memory_fraction = 0.01 # 80MB memory is enough to run on TX2
        """ """
        
        """ """ """ """ """ """ """ """ """ """ """
        PREPARE GRAPH I/O TO VARIABLE
        """ """ """ """ """ """ """ """ """ """ """
        # Define Input and Ouput tensors
        image_tensor = graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = graph.get_tensor_by_name('detection_scores:0')
        detection_classes = graph.get_tensor_by_name('detection_classes:0')
        

        """ """ """ """ """ """ """ """ """ """ """
        START WORKER THREAD
        """ """ """ """ """ """ """ """ """ """ """
        # gpu_worker uses in split_model and non-split_model
        gpu_tag = 'GPU'
        cpu_tag = 'CPU'
        gpu_worker = SessionWorker(gpu_tag, graph, config)
        gpu_opts = [detection_boxes, detection_scores, detection_classes]
        """ """

        """
        START VISUALIZE WORKER
        """
        if VISUALIZE and VIS_WORKER:
            q_out = Queue.Queue()
            vis_worker = MPVisualizeWorker(cfg, MPVariable.vis_in_con)
            """ """ """ """ """ """ """ """ """ """ """
            START SENDER THREAD
            """ """ """ """ """ """ """ """ """ """ """
            start_sender(MPVariable.det_out_con, q_out)
        proc_frame_counter = 0
        vis_proc_time = 0

        if VISUALIZE:
            cv2.namedWindow("Object Detection", 0)#The 0 represents the window type
            cv2.resizeWindow("Object Detection", WIDTH, HEIGHT)
        
        """ """ """ """ """ """ """ """ """ """ """
        START FRAME ACQUISITION
        """ """ """ """ """ """ """ """ """ """ """

        from lib.video import VideoReader
        video_reader = VideoReader()

        video_reader.start(VIDEO_INPUT, WIDTH, HEIGHT, save_to_file=SAVE_TO_FILE)
        frame_cols, frame_rows = video_reader.getSize()
        """ STATISTICS FONT """
        fontScale = frame_rows/1000.0
        if fontScale < 0.4:
            fontScale = 0.4
        fontThickness = 1 + int(fontScale)
        fontFace = cv2.FONT_HERSHEY_SIMPLEX

        dir_path, filename = os.path.split(VIDEO_INPUT)
        filepath_prefix = filename

        """ """


        """ """ """ """ """ """ """ """ """ """ """
        DETECTION LOOP
        """ """ """ """ """ """ """ """ """ """ """
        print('Starting Detection')
        pr = PowerRead(device = DEVICE.value)
        sleep_interval = 0.005
        top_in_time = None
        frame_in_processing_counter = 0
        start_time = time.time()
        try:
            if not video_reader.running:
                raise IOError(("Input src error."))
            while MPVariable.running.value:
                if top_in_time is None:
                    top_in_time = time.time()
                """
                SPRIT/NON-SPLIT MODEL CAMERA TO WORKER
                """
                if video_reader.running:
                    if gpu_worker.is_sess_empty(): # must need for speed
                        cap_in_time = time.time()
                        if SRC_FROM == IMAGE:
                            frame, filepath = video_reader.read()
                            if frame is not None:
                                frame_in_processing_counter += 1
                        else:
                            frame = video_reader.read()
                            if frame is not None:
                                filepath = filepath_prefix+'_'+str(proc_frame_counter)+'.png'
                                frame_in_processing_counter += 1
                        if frame is not None:
                            image_expanded = np.expand_dims(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), axis=0) # np.expand_dims is faster than []
                            #image_expanded = np.expand_dims(frame, axis=0) # BGR image for input. Of couse, bad accuracy in RGB trained model, but speed up.
                            cap_out_time = time.time()
                            # put new queue
                            gpu_feeds = {image_tensor: image_expanded}
                            gpu_extras = {'image':frame, 'top_in_time':top_in_time, 'cap_in_time':cap_in_time, 'cap_out_time':cap_out_time, 'filepath': filepath} # always image draw.
                            gpu_worker.put_sess_queue(gpu_opts, gpu_feeds, gpu_extras)
                elif frame_in_processing_counter <= 0:
                    MPVariable.running.value = False
                    break

                g = gpu_worker.get_result_queue()
                if SPLIT_MODEL:
                    # if g is None: gpu thread has no output queue. ok skip, let's check cpu thread.
                    if g is not None:
                        # gpu thread has output queue.
                        result_slice_out, extras = g['results'], g['extras']
                        if cpu_worker.is_sess_empty():
                            # When cpu thread has no next queue, put new queue.
                            # else, drop gpu queue.
                            cpu_feeds = {}
                            for i in range(len(result_slice_out)):
                                cpu_feeds.update({split_in[i]:result_slice_out[i]})
                            cpu_extras = extras
                            cpu_worker.put_sess_queue(cpu_opts, cpu_feeds, cpu_extras)
                        else:
                            # else: cpu thread is busy. don't put new queue. let's check cpu result queue.
                            frame_in_processing_counter -= 1
                    # check cpu thread.
                    q = cpu_worker.get_result_queue()
                else:
                    """
                    NON-SPLIT MODEL
                    """
                    q = g
                if q is None:
                    """
                    SPLIT/NON-SPLIT MODEL
                    """
                    # detection is not complete yet. ok nothing to do.
                    time.sleep(sleep_interval)
                    continue

                frame_in_processing_counter -= 1
                boxes, scores, classes, extras = q['results'][0], q['results'][1], q['results'][2], q['extras']
                boxes, scores, classes = np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes)
                det_out_time = time.time()

                """
                ALWAYS BOX DRAW ON IMAGE
                """
                vis_in_time = time.time()
                image = extras['image']
                if SRC_FROM == IMAGE:
                    filepath = extras['filepath']
                    frame_rows, frame_cols = image.shape[:2]
                    """ STATISTICS FONT """
                    fontScale = frame_rows/1000.0
                    if fontScale < 0.4:
                        fontScale = 0.4
                    fontThickness = 1 + int(fontScale)
                else:
                    filepath = extras['filepath']
                image = visualization(category_index, image, boxes, scores, classes, DEBUG_MODE, VIS_TEXT, FPS_INTERVAL,
                                      fontFace=fontFace, fontScale=fontScale, fontThickness=fontThickness)

                """
                VISUALIZATION
                """
                if VISUALIZE:
                    if (MPVariable.vis_skip_rate.value == 0) or (proc_frame_counter % MPVariable.vis_skip_rate.value < 1):
                        if VIS_WORKER:
                            q_out.put({'image':image, 'vis_in_time':vis_in_time})
                        else:
                            """
                            SHOW
                            """
                            cv2.imshow("Object Detection", image)
                            # Press q to quit
                            if cv2.waitKey(1) & 0xFF == 113: #ord('q'):
                                break
                            MPVariable.vis_frame_counter.value += 1
                            vis_out_time = time.time()
                            """
                            PROCESSING TIME
                            """
                            vis_proc_time = vis_out_time - vis_in_time
                            MPVariable.vis_proc_time.value += vis_proc_time
                else:
                    """
                    NO VISUALIZE
                    """
                    for box, score, _class in zip(boxes, scores, classes):
                        if proc_frame_counter % DET_INTERVAL == 0 and score > DET_TH:
                            label = category_index[_class]['name']
                            print("label: {}\nscore: {}\nbox: {}".format(label, score, box))

                    vis_out_time = time.time()
                    """
                    PROCESSING TIME
                    """
                    vis_proc_time = vis_out_time - vis_in_time

                if SAVE_TO_FILE:
                    if SRC_FROM == IMAGE:
                        video_reader.save(image, filepath)
                    else:
                        video_reader.save(image)

                proc_frame_counter += 1
                if proc_frame_counter > 100000:
                    proc_frame_counter = 0
                """
                PROCESSING TIME
                """
                top_in_time = extras['top_in_time']
                cap_proc_time = extras['cap_out_time'] - extras['cap_in_time']
                gpu_proc_time = extras[gpu_tag+'_out_time'] - extras[gpu_tag+'_in_time']
                if SPLIT_MODEL:
                    cpu_proc_time = extras[cpu_tag+'_out_time'] - extras[cpu_tag+'_in_time']
                else:
                    cpu_proc_time = 0
                lost_proc_time = det_out_time - top_in_time - cap_proc_time - gpu_proc_time - cpu_proc_time
                total_proc_time = det_out_time - top_in_time
                MPVariable.cap_proc_time.value += cap_proc_time
                MPVariable.gpu_proc_time.value += gpu_proc_time
                MPVariable.cpu_proc_time.value += cpu_proc_time
                MPVariable.lost_proc_time.value += lost_proc_time
                MPVariable.total_proc_time.value += total_proc_time
                
                with open("Processing_Times.txt", "a") as f:
                    f.write("{:f} {:f} {:f}\n".format(gpu_proc_time, vis_proc_time, total_proc_time))
                    f.close()
                if DEBUG_MODE:
                    if SPLIT_MODEL:
                        sys.stdout.write('snapshot FPS:{: ^5.1f} total:{: ^10.5f} cap:{: ^10.5f} gpu:{: ^10.5f} cpu:{: ^10.5f} lost:{: ^10.5f} | vis:{: ^10.5f}\n'.format(
                            MPVariable.fps.value, total_proc_time, cap_proc_time, gpu_proc_time, cpu_proc_time, lost_proc_time, vis_proc_time))
                    else:
                        sys.stdout.write('snapshot FPS:{: ^5.1f} total:{: ^10.5f} cap:{: ^10.5f} gpu:{: ^10.5f} lost:{: ^10.5f} | vis:{: ^10.5f}\n'.format(
                            MPVariable.fps.value, total_proc_time, cap_proc_time, gpu_proc_time, lost_proc_time, vis_proc_time))
                """
                EXIT WITHOUT GUI
                """
                if not VISUALIZE and MAX_FRAMES > 0:
                    if proc_frame_counter >= MAX_FRAMES:
                        MPVariable.running.value = False
                        break

                """
                CHANGE SLEEP INTERVAL
                """
                if MPVariable.frame_counter.value == 0 and MPVariable.fps.value > 0:
                    sleep_interval = 0.1 / MPVariable.fps.value
                    MPVariable.sleep_interval.value = sleep_interval
                MPVariable.frame_counter.value += 1
                top_in_time = None
            """
            END while
            """
        except KeyboardInterrupt:
            pass
        except:
            import traceback
            traceback.print_exc()
        finally:
            end_time = time.time() - start_time
            print("Total Time to Inference and Process Image: {:.3f} s".format(end_time))
            """ """ """ """ """ """ """ """ """ """ """
            CLOSE
            """ """ """ """ """ """ """ """ """ """ """
            if VISUALIZE and VIS_WORKER:
                q_out.put(None)
            MPVariable.running.value = False
            gpu_worker.stop()
            pr.stop()
            if SPLIT_MODEL:
                cpu_worker.stop()
            video_reader.stop()

            if VISUALIZE:
                cv2.destroyAllWindows()
            num_gpu = 1
            pw = []
            for n in range(num_gpu):
                with open("Power_Measurements_{}.txt".format(n), "r") as f:
                    pw += [[float(i) / 1000 for i in f]]
                    f.close()
                
            with open("Processing_Times.txt") as f:
                next(f)
                tm = [[float(x)*1000 for x in i.split()] for i in f]
                f.close()
            tm = np.array(tm)
            pw = np.array(pw)
            import matplotlib.pyplot as plt
            import matplotlib as mpl
#            from math import sqrt
            mpl.use('tkagg')
#            x = np.arange(0,len(pw))
            if DEVICE.value in (0, 1):
                pwr = []
                for n in range(num_gpu):
                    pwr += [[ x for x in pw[n] if 65 <= x <= 180]]
            elif DEVICE.value == 1:
                pwr = pw
            
            for n in range(num_gpu):
                #Power Measurements Plot
                fig, ax = plt.subplots()
    #            plt.stem(x, pw, use_line_collection = True)
                plt.ylabel('Frequency')
                plt.xlabel('GPU Power (W)')
                _, b, _ = plt.hist(pwr[n], bins = 'auto', edgecolor = 'black', 
                                   linewidth = 0.75)
    #            plt.xlabel('GPU Power (W)')
    #            plt.ylabel('Number of Times')
                plt.title('Power Consumed by GPU_{} During Inferencing'.format(n))
                ax.set_axisbelow(True)
                
                ax.minorticks_on()
                ax.grid(which = 'major', linestyle = '-',
                         linewidth = 0.5, color = 'black')
                ax.grid(which = 'minor', linestyle = ':',
                         linewidth = 0.5, color = 'black')
                print("Mean GPU_{} Consumption: {:.3f} W".format(n,np.mean(pwr)))
                print("Standard GPU_{} Consumption Deviation: {:.3f} W".format(n,np.std(pwr)))
            #Worker Processing time plot
            fig, ax = plt.subplots()
            ax.set_axisbelow(True)
            
            ax.minorticks_on()
            ax.grid(which = 'major', linestyle = '-',
                     linewidth = 0.5, color = 'black')
            ax.grid(which = 'minor', linestyle = ':',
                     linewidth = 0.5, color = 'black')
#            plt.stem(np.arange(0, len(tm[:,0])), tm[:,0], 
#                     use_line_collection = True, 
#                     label = 'Worker Processing Time')
#            plt.stem(np.arange(0, len(tm[:,0])), tm[:,1], 
#                     use_line_collection = True, 
#                     label = 'Visualization Processing Time',
#                     linefmt = 'C1-', markerfmt = 'C1o')
#            plt.stem(np.arange(0, len(tm[:,0])), tm[:,2], 
#                     use_line_collection = True, 
#                     label = 'Total Processing Time',
#                     linefmt = 'C2-', markerfmt = 'C2o')
            
            plt.ylabel('Frequency')
            plt.xlabel('Worker Time (ms)')
            plt.hist(tm[:,0], bins = 'auto', label = 'Worker Processing Time', 
                     edgecolor = 'black', linewidth = 0.75)
            plt.legend(loc = 'best')
            plt.title('Time taken to inference image')
            plt.grid(True, which = 'both', axis = 'both', zorder = 0)
            print("Mean Worker Time: {:.3f} ms".format(np.mean(tm[1:,0])))
            print("Standard Worker Deviation: {:.3f} ms".format(np.std(tm[1:,0])))
            
            plt.show()

            from numba import cuda
            cuda.select_device(0)
            cuda.close()
            """ """

        return
