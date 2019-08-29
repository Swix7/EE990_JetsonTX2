from tensorflow.core.framework import graph_pb2
import copy

import os
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
from tf_trt_models.detection import download_detection_model
from tf_trt_models.detection import build_detection_graph
from tf_trt_models.detection import get_output_names



class LoadFrozenGraph():
    """
    LOAD FROZEN GRAPH
    TRT Graph
    """
    def __init__(self, cfg):
        self.cfg = cfg
        return

    def load_graph(self):
        print('Building Graph')
        trt_graph_def=self.build_trt_graph()

        # force CPU device placement for NMS ops
        for node in trt_graph_def.node:
            if 'NonMaxSuppression' in node.name:
                node.device = '/device:CPU:0'
            else:
                node.device = '/device:GPU:0'
        return self.non_split_trt_graph(graph_def=trt_graph_def)


    def print_graph(self, graph):
        """
        PRINT GRAPH OPERATIONS
        """
        print("{:-^32}".format(" operations in graph "))
        for op in graph.get_operations():
            print(op.name,op.outputs)
        return

    def print_graph_def(self, graph_def):
        """
        PRINT GRAPHDEF NODE NAMES
        """
        print("{:-^32}".format(" nodes in graph_def "))
        for node in graph_def.node:
            print(node)
        return

    def print_graph_operation_by_name(self, graph, name):
        """
        PRINT GRAPH OPERATION DETAILS
        """
        op = graph.get_operation_by_name(name=name)
        print("{:-^32}".format(" operations in graph "))
        print("{:-^32}\n{}".format(" op ", op))
        print("{:-^32}\n{}".format(" op.name ", op.name))
        print("{:-^32}\n{}".format(" op.outputs ", op.outputs))
        print("{:-^32}\n{}".format(" op.inputs ", op.inputs))
        print("{:-^32}\n{}".format(" op.device ", op.device))
        print("{:-^32}\n{}".format(" op.graph ", op.graph))
        print("{:-^32}\n{}".format(" op.values ", op.values()))
        print("{:-^32}\n{}".format(" op.op_def ", op.op_def))
        print("{:-^32}\n{}".format(" op.colocation_groups ", op.colocation_groups))
        print("{:-^32}\n{}".format(" op.get_attr ", op.get_attr("T")))
        i = 0
        for output in op.outputs:
            op_tensor = output
            tensor_shape = op_tensor.get_shape().as_list()
            print("{:-^32}\n{}".format(" outputs["+str(i)+"] shape ", tensor_shape))
            i += 1
        return

    # helper function for split model
    def node_name(self, n):
        if n.startswith("^"):
            return n[1:]
        else:
            return n.split(":")[0]

    def non_split_trt_graph(self, graph_def):
        tf.import_graph_def(graph_def, name='')
        return tf.get_default_graph()

    def build_trt_graph(self):
        MODEL             = self.cfg['model']
        PRECISION_MODE    = self.cfg['precision_model']
        CONFIG_FILE       = "data/" + MODEL + '.config'   # ./data/ssd_inception_v2_coco.config 
        CHECKPOINT_FILE   = 'data/' + MODEL + '/model.ckpt'    # ./data/ssd_inception_v2_coco/model.ckpt
        FROZEN_MODEL_NAME = MODEL+'_trt_' + PRECISION_MODE + '.pb'
        TRT_MODEL_DIR     = 'data'
        LOGDIR            = 'logs/' + MODEL + '_trt_' + PRECISION_MODE

        if os.path.exists(os.path.join(TRT_MODEL_DIR, FROZEN_MODEL_NAME)) is False:
            config_path, checkpoint_path = download_detection_model(MODEL, 'data')
    
            frozen_graph_def, _, _ = build_detection_graph(
                config=config_path,
                checkpoint=checkpoint_path,
                score_threshold = 0.5,
                force_nms_cpu = False
                
            )
    
            tf.reset_default_graph()
            trt_graph_def = trt.create_inference_graph(
                input_graph_def=frozen_graph_def,
                outputs=get_output_names(MODEL),
                max_batch_size=1,
                max_workspace_size_bytes=1<<30,
                precision_mode=PRECISION_MODE,
                minimum_segment_size=50
            )
#            tf.train.write_graph(trt_graph_def, TRT_MODEL_DIR,
#                                 FROZEN_MODEL_NAME, as_text=False)
#    
#            train_writer = tf.summary.FileWriter(LOGDIR)
#            train_writer.add_graph(tf.get_default_graph())
#            train_writer.flush()
#            train_writer.close()
            with open(os.path.join(TRT_MODEL_DIR, FROZEN_MODEL_NAME), 'wb') as f:
                f.write(trt_graph_def.SerializeToString())
        else:
            print("It Works")
            trt_graph_def = tf.GraphDef()
            with tf.gfile.GFile(os.path.join(TRT_MODEL_DIR, FROZEN_MODEL_NAME), 'rb') as f:
                trt_graph_def.ParseFromString(f.read())
        

        return trt_graph_def
