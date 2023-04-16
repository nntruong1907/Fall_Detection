import cv2
import numpy as np
import tensorflow as tf
import utils_pose as utils
from movenet import Movenet
from data import BodyPart
import os

from def_lib import detect, get_keypoint_landmarks, landmarks_to_embedding, draw_prediction_on_image, predict_pose, draw_class_on_image, write_video

path = './videos/office1_test_fall.mp4'

def gen_video(path):
    model = tf.keras.models.load_model("./models/model_fall.h5")
    cap = cv2.VideoCapture(path)
    time_step = 5
    label = "waiting"
    i = 0
    lm = []
    list = []

    while cap.isOpened():
        ret, frame = cap.read()
        # Reshape Image
        if ret == True:
            img = frame.copy()
            img = cv2.resize(img, (854, 480))
#             img = cv2.resize(img, (640, 360))
            img = tf.convert_to_tensor(img, dtype=tf.uint8)
            i = i + 1

            print(f"Start detect: frame {i}")
            person = detect(img)
            landmarks = get_keypoint_landmarks(person)
            lm_pose = landmarks_to_embedding(tf.reshape(
                tf.convert_to_tensor(landmarks), (1, 51)))
    #         print(lm_pose)
            lm.append(lm_pose)
            img = np.array(img)
            img = draw_prediction_on_image(img, person, crop_region=None,
                                           close_figure=False, keep_input_size=True)
            if (len(lm) == time_step):
                lm = tf.reshape(lm, (1, 34, 5))

                label = predict_pose(model, lm, label)
                lm = []

            img = np.array(img)
            img = draw_class_on_image(label, img)
            list.append(img)

            cv2.imshow('Fall Detection', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # cv2.destroyAllWindows()

        else:
            break

    cap.release()
    # out_path = os.path.join('./outputs/', path)
    # write_video(out_path, np.array(list), 24)

gen_video(path)
