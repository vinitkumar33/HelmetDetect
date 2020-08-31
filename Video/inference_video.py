import os
import tensorflow as tf
import numpy as np
import cv2 as cv
from PIL import Image
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops

class DefectDetection(object):
 
    def show_inference(self, model, frame):
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      image_np = np.array(frame)
      PATH_TO_LABELS = 'helmets_label_map.pbtxt'
      # Actual detection.
      output_dict = self.run_inference_for_single_image(model, image_np)
      label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
      categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=2, use_display_name=True)
      category_index = label_map_util.create_category_index(categories)
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          output_dict['detection_boxes'],
          output_dict['detection_classes'],
          output_dict['detection_scores'],
          category_index,
          instance_masks=output_dict.get('detection_masks_reframed', None),
          use_normalized_coordinates=True,
          line_thickness=8)

      #display(Image.fromarray(image_np))
      cv.imshow('window',image_np)
      if cv.waitKey(1) & 0xFF == ord('q'): 
        return

    def run_inference_for_single_image(self, model, image):
      image = np.asarray(image)
      # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
      input_tensor = tf.convert_to_tensor(image)
      # The model expects a batch of images, so add an axis with `tf.newaxis`.
      input_tensor = input_tensor[tf.newaxis,...]

      # Run inference
      output_dict = model(input_tensor)

      # All outputs are batches tensors.
      # Convert to numpy arrays, and take index [0] to remove the batch dimension.
      # We're only interested in the first num_detections.
      num_detections = int(output_dict.pop('num_detections'))
      output_dict = {key:value[0, :num_detections].numpy() 
                     for key,value in output_dict.items()}
      output_dict['num_detections'] = num_detections

      # detection_classes should be ints.
      output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
       
      # Handle models with masks:
      if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                  output_dict['detection_masks'], output_dict['detection_boxes'],
                   image.shape[0], image.shape[1])      
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.95,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
        
      return output_dict

def load_model(model_name):
    model_dir = "model"

    #localhost_save_option = tf.saved_model.SaveOptions(experimental_io_device="/job:localhost")
    #model.save(model_dir, options=localhost_save_option)

    # Restore the weights
    #model = tf.keras.models.load_model(model_dir, options=localhost_save_option)
    #model_dir = '/Users/i351150/Downloads/model'

    localhost_load_option=tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
    #localhost_save_option = tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')
    model = tf.saved_model.load(model_dir, tags=None, options=localhost_load_option)

    #model = tf.keras.models.load_model(model_dir, custom_objects=None, compile=True, options=None)
    #model.save(model_dir, options=localhost_save_option)
    #model = model.signatures['serving_default']

    return model

if __name__ == "__main__":
    classifier = DefectDetection()
    model_name = 'frozen_inference_graph.pb'
    detection_model = load_model(model_name)
    vid = cv.VideoCapture("videoplayback.mp4");
    while(True):
      ret, frame = vid.read() 
      # Display the resulting frame 
      cv.imshow('frame', frame) 
      classifier.show_inference(detection_model, frame)
    vid.release() 
    cv.destroyAllWindows()