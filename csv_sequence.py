from os import listdir
import os
from datetime import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
from def_lib import detect, get_keypoint_landmarks, landmarks_to_embedding, write_to_csv, fusion_all_csv

"""
Link dataset image (9GB): https://mega.nz/file/FVQTRZgC#XZM1jQjEOGZ2977Dr6rzS7ifkHAWiA1ewSIn8m2o7_Y
Link dataset video (1GB): https://mega.nz/file/JV5B0IDD#FETP9vLLbw1PhWxpbz31SIumEkruNrF2lt4CRG4lDTo
"""

start = datetime.now()

def save_data(raw_folder):

    print("Bắt đầu xử lý ảnh...")
    pixels = []
    labels = []
    class_names = []
    tmp = 0
    
    sequence = 5
    
    for folder_class in os.listdir(raw_folder):
        
        for folder_file in os.listdir(raw_folder + folder_class):
            
            if folder_class!='.DS_Store':
                print("folder_class = ", folder_class)
                
                for i in range(sequence-1, len(os.listdir(raw_folder + folder_class +"/"+ folder_file))):

                    if os.listdir(raw_folder + folder_class +"/"+ folder_file)[i]!='.DS_Store':
                        print("File = ", os.listdir(raw_folder + folder_class +"/"+ folder_file)[i] + " Folder " + str(folder_class))
                        lm = []
                        for j in range(1, sequence + 1):
                            
                            img_path = raw_folder + folder_class +"/"+ folder_file +"/"+ listdir(raw_folder + folder_class +"/"+ folder_file)[i-(sequence-j)]
                            print("img path: ", img_path)
                            image = tf.io.read_file(img_path)
                            image = tf.io.decode_jpeg(image)
                            person = detect(image)
                            pose_landmarks = get_keypoint_landmarks(person)
                            lm_pose = landmarks_to_embedding(tf.reshape(tf.convert_to_tensor(pose_landmarks), (1, 51)))

                            lm.append(lm_pose)
                        
                        print("lm", np.array(lm).shape)

                        pixels.append(lm)
                        labels.append(tmp)
                        class_names.append(folder_class)
                        lm = []
        
#         write_to_csv("./data/csv_train/" + folder_class + ".csv", pixels, labels, class_names)
        write_to_csv("./data/csv_test/" + folder_class + ".csv", pixels, labels, class_names)

        tmp += 1
        pixels = []
        labels = []
        class_names = []

# save_data("./dataset/train/")
# fusion_all_csv("./data/csv_train/", './dataset/train/',"./data/train_5f.csv")
# duration_train = datetime.now() - start
# makeTrainCSV_eval = 'TRAIN.CSV CREATION COMPLETE TIME: ' + str(duration_train)
# print(makeTrainCSV_eval)
# train_path = './statistics/timeCompletedTrainCSV.txt'
# with open(train_path, mode='w') as f:
#     f.writelines(makeTrainCSV_eval)
# f.close()

save_data("./dataset/test/")
fusion_all_csv("./data/csv_test/", './dataset/test/',"./data/test_5f.csv")
duration_test = datetime.now() - start
makeTestCSV_eval = 'TEST.CSV CREATION COMPLETE TIME: ' + str(duration_test)
print(makeTestCSV_eval)
test_path = './statistics/timeCompletedTestCSV.txt'
with open(test_path, mode='w') as f:
    f.writelines(makeTestCSV_eval)
f.close()