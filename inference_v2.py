import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import cv2 as cv

class DefectDetection:

    def __init__(self):

        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        PATH_TO_CKPT = 'saved_model.pb'

        self.detection_graph = tf.Graph()
        self.confidence = .50

        with tf.io.gfile.GFile(PATH_TO_CKPT, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="")
        #with self.detection_graph.as_default():
        #    od_graph_def = tf.compat.v1.GraphDef()
        #    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        #        serialized_graph = fid.read()
        #        od_graph_def.ParseFromString(serialized_graph)
        #        tf.import_graph_def(od_graph_def, name='')

    #def load_model():
    #    with tf.gfile.GFile("saved_model.pb", "rb") as f:
    #        graph_def = tf.GraphDef()
    #        graph_def.ParseFromString(f.read())
#
    #    with tf.Graph().as_default() as graph:
    #        tf.import_graph_def(graph_def, name="")
    #    return graph
#
    def detect(self, img_np, img_box):

        detection_graph = self.detection_graph
        # labels = []
        label = None
        with detection_graph.as_default():
            with tf.compat.v1.Session(graph=detection_graph) as sess:

                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                img_expanded = np.expand_dims(img_np, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: img_expanded})

                box = tuple(boxes[0].tolist())
                for score_seq, score in enumerate(np.nditer(scores[0])):
                    if score < self.confidence:
                        break
                    print(score)
                    # labels.append(int(classes[0][score_seq]))
                    label = int(classes[0][score_seq])
                    ymin, xmin, ymax, xmax = tuple(box[score_seq])
                    im_height, im_width = img_box.shape[:2]
                    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                                  ymin * im_height, ymax * im_height)
                    cv.rectangle(img_box, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 4)

        return (label, img_box)

if __name__ == "__main__":
    classifier = DefectDetection()
    img_box = cv.imread('image1.jpeg')
    img_np = cv.cvtColor(img_box, cv.COLOR_RGB2BGR)
    labels, img_box = classifier.detect(img_np, img_box)
    cv.imwrite("image.jpg", img_box)