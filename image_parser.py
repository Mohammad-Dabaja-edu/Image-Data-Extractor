import os
import tensorflow as tf
import time
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import numpy as np
from PIL import Image
import warnings
import cv2

# warnings.filterwarnings('ignore')  # Suppress Matplotlib warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1)
# tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logging (2)


PATH_TO_MODEL_DIR = "model/my_model_gray_9"
PATH_TO_LABELS = "model/label_map.pbtxt"
PATH_TO_IMAGES_DIR = "static"


class ImageParser:

    def __init__(self):

        # Enable GPU dynamic memory allocation
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            print(f"setting{gpu}")
            tf.config.experimental.set_memory_growth(gpu, True)

        # Load the model
        PATH_TO_CFG = PATH_TO_MODEL_DIR + "\pipeline.config"
        PATH_TO_CKPT = PATH_TO_MODEL_DIR + "\checkpoint"

        print('Loading model... ', end='')
        start_time = time.time()

        # Load pipeline config and build a detection model
        configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
        model_config = configs['model']
        self.detection_model = model_builder.build(model_config=model_config, is_training=False)

        # Restore checkpoint
        ckpt = tf.compat.v2.train.Checkpoint(model=self.detection_model)
        ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()

        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Done! Took {} seconds'.format(elapsed_time))

        # Load label map data (for plotting)
        self.category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                                 use_display_name=True)

    # @tf.function
    def detect_fn(self, image):
        """Detect objects in image."""

        start_time = time.time()
        image, shapes = self.detection_model.preprocess(image)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print('preprocess! Took {} seconds'.format(elapsed_time))

        start_time = time.time()
        prediction_dict = self.detection_model.predict(image, shapes)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print('predict! Took {} seconds'.format(elapsed_time))

        start_time = time.time()
        detections = self.detection_model.postprocess(prediction_dict, shapes)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print('postprocess! Took {} seconds'.format(elapsed_time))
        return detections

    # def load_image_into_numpy_array(self, path):
    #     return np.array(Image.open(path))

    def load_image_into_numpy_array(self, path):
        # The function supports only grayscale images
        image = Image.open(path)

        if np.array(image).shape[2] == 1:
            last_axis = -1
            dim_to_repeat = 2
            repeats = 3
            grscale_img_3dims = np.expand_dims(image, last_axis)
            training_image = np.repeat(grscale_img_3dims, repeats, dim_to_repeat).astype('uint8')
            assert len(training_image.shape) == 3
            assert training_image.shape[-1] == 3
            return training_image
        else:
            return np.array(image)

    def parse(self, filename, parses_dir):
        image_path = os.path.join(PATH_TO_IMAGES_DIR, filename.split('.')[0], filename)
        print('Running inference for {}... '.format(image_path), end='')
        image_np = self.load_image_into_numpy_array(image_path)
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)

        detections = self.detect_fn(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'] + label_id_offset,
            detections['detection_scores'],
            self.category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.10,
            agnostic_mode=False)

        ###########################################

        output_directory = parses_dir
        image_height, image_width, _ = image_np.shape
        # get label and coordinates of detected objects
        output = dict()
        for index, score in enumerate(detections['detection_scores']):
            label = self.category_index[detections['detection_classes'][index] + 1]['name']
            ymin, xmin, ymax, xmax = detections['detection_boxes'][index]
            if score > 0.1:
                if label in output:
                    if score > output[label][0]:
                        output[label] = (
                            score, int(xmin * image_width), int(ymin * image_height), int(xmax * image_width),
                            int(ymax * image_height))
                else:
                    output[label] = (score, int(xmin * image_width), int(ymin * image_height), int(xmax * image_width),
                                     int(ymax * image_height))

        i = 0
        # Save images and labels
        for l in output:
            _, x_min, y_min, x_max, y_max = output[l]
            array = cv2.cvtColor(np.array(image_np), cv2.COLOR_RGB2BGR)
            image = Image.fromarray(array)
            cropped_img = image.crop((x_min, y_min, x_max, y_max))
            file_path = f'{output_directory}/{l}.jpg'
            cropped_img.save(file_path, "JPEG", icc_profile=cropped_img.info.get('icc_profile'))
            i = i + 1

        ########################################

        print('Done')
