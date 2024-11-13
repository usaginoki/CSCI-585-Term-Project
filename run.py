import os
from typing import Any
# import multiprocessing

# multiprocessing.set_start_method("spawn")


# Define the DictValue replacement before importing cv2
def get_dict_value() -> Any:
    try:
        import cv2

        return cv2.dnn.DictValue
    except AttributeError:
        # If DictValue is not available, create a simple replacement
        class DictValueReplacement:
            def __init__(self, val):
                self.val = val

            def __repr__(self):
                return str(self.val)

        return DictValueReplacement


# Now import cv2 and patch it
import cv2

cv2.dnn.DictValue = get_dict_value()

# Continue with your other imports
import source.audio_analysis_utils.model as audio_model
import source.audio_analysis_utils.predict as audio_predict

import source.face_emotion_utils.model as face_model
import source.face_emotion_utils.predict as face_predict
import source.face_emotion_utils.utils as face_utils
import source.config as config
import source.face_emotion_utils.preprocess_main as face_preprocess_main

import source.audio_face_combined.model as combined_model
import source.audio_face_combined.preprocess_main as combined_data
import source.audio_face_combined.combined_config as combined_config
import source.audio_face_combined.predict as combined_predict
import source.audio_face_combined.download_video as download_youtube
import source.audio_face_combined.utils as combined_utils

import source.audio_face_transformer.model as transformer_model
import source.audio_face_transformer.predict as transformer_predict
import source.audio_face_transformer.tr_config as transformer_config
import source.audio_face_transformer.preprocess_main as transformer_preprocess
import source.audio_face_transformer.utils as transformer_utils
import source.audio_face_transformer.download_video as transformer_download

import sys


def get_dict_value() -> Any:
    try:
        return cv2.dnn.DictValue
    except AttributeError:
        # If DictValue is not available, create a simple replacement
        class DictValueReplacement:
            def __init__(self, val):
                self.val = val

            def __repr__(self):
                return str(self.val)

        return DictValueReplacement


cv2.dnn.DictValue = get_dict_value()


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("1) Audio analysis")
        print("2) Face analysis")
        print("3) AV combined analysis")
        print("4) Transformer analysis")
        ch = int(input("Which model: "))
    else:
        ch = int(sys.argv[1])

    if ch == 1:
        if len(sys.argv) <= 2:
            ch = int(
                input(
                    "1) Tune hyperparameters\n"
                    "2) Train model\n"
                    "3) Preprocess data\n"
                    "4) Predict emotion\n"
                    "5) Test on test set\n"
                    "\n\nEnter your choice: "
                )
            )
        else:
            ch = int(sys.argv[2])

        if ch == 1:
            if len(sys.argv) > 2:
                ch = int(sys.argv[3])
            else:
                ch = int(input("Continue tuning hyperparameters? (1/0): "))
            audio_model.hyper_parameter_optimise(load_if_exists=ch)
        elif ch == 2:
            audio_model.train_using_best_values()
        elif ch == 3:
            audio_model.train_using_best_values(preprocess_again=True)
        elif ch == 4:
            file_name = str(input("Enter file name: "))
            audio_predict.predict(file_name)
        elif ch == 5:
            audio_model.train_model(preprocess_again=False, test_on_test_set=True)

    elif ch == 2:
        if len(sys.argv) <= 2:
            ch = int(
                input(
                    "1) Tune hyperparameters\n"
                    "2) Train model\n"
                    "3) Preprocess data\n"
                    "4) Predict emotion\n"
                    "5) Predict from video\n"
                    "6) Predict from webcam\n"
                    "7) Predict from test set\n"
                    "\n\nEnter your choice: "
                )
            )
        else:
            ch = int(sys.argv[2])

        if ch == 1:
            if len(sys.argv) > 2:
                ch = int(sys.argv[3])
            else:
                ch = int(input("Continue tuning hyperparameters? (1/0): "))
            face_model.hyper_parameter_optimise(load_if_exists=ch)
        elif ch == 2:
            face_model.train_using_best_values()
        elif ch == 3:
            face_preprocess_main.preprocess_images(
                original_images_folders=config.ALL_EXTRACTED_FACES_FOLDERS,
                output_path=config.PREPROCESSED_IMAGES_FOLDER_PATH,
                print_flag=True,
            )
        elif ch == 4:
            file_name = str(input("Enter file name: "))
            file_name = face_utils.find_filename_match(
                known_filename=file_name, directory=config.INPUT_FOLDER_PATH
            )
            print("file_name", file_name)
            image = cv2.imread(file_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_predict.predict(image)
        elif ch == 5:
            file_name = str(input("Enter file name: "))
            file_name = face_utils.find_filename_match(
                known_filename=file_name, directory=config.INPUT_FOLDER_PATH
            )
            print("file_name", file_name)
            face_predict.predict(file_name, video_mode=True)
        elif ch == 6:
            face_predict.predict(webcam_mode=True)
        elif ch == 7:
            face_model.train_model(test_on_testset=True)

    elif ch == 3:
        if len(sys.argv) <= 2:
            ch = int(
                input(
                    "1) Tune hyperparameters\n"
                    "2) Train model\n"
                    "3) Preprocess data\n"
                    "4) Predict emotion\n"
                    "5) Predict from test set\n"
                    "6) Predict from YouTube video\n"
                    "7) Visualise video predictions\n"
                    "\n\nEnter your choice: "
                )
            )
        else:
            ch = int(sys.argv[2])

        if ch == 1:
            if len(sys.argv) > 2:
                ch = int(sys.argv[3])
            else:
                ch = int(input("Continue tuning hyperparameters? (1/0): "))
            combined_model.hyper_parameter_optimise(load_if_exists=ch)
        elif ch == 2:
            combined_model.train_using_best_values()
        elif ch == 3:
            best_hyperparameters = face_utils.load_dict_from_json(
                config.AUDIO_BEST_HP_JSON_SAVE_PATH
            )
            print(f"Best hyperparameters, {best_hyperparameters}")

            N_FFT = best_hyperparameters["N_FFT"]
            HOP_LENGTH = best_hyperparameters["HOP_LENGTH"]
            NUM_MFCC = best_hyperparameters["NUM_MFCC"]
            combined_data.preprocess_videos(
                N_FFT=combined_config.tune_hp_ranges["N_FFT"][0][0],
                NUM_MFCC=combined_config.tune_hp_ranges["NUM_MFCC"][0][0],
                HOP_LENGTH=combined_config.tune_hp_ranges["HOP_LENGTH"][0][0],
                original_videos_folder=config.ALL_EXTRACTED_AV_FOLDERS,
                output_path=config.PREPROCESSED_AV_FOLDER_PATH,
            )
        elif ch == 4:
            video_name = str(input("Enter video name in input path: "))
            video_name = face_utils.find_filename_match(
                video_name, config.INPUT_FOLDER_PATH
            )
            print("video_path", video_name)
            combined_predict.predict_video(video_name)
        elif ch == 5:
            combined_model.train_model(test_on_test_set=True)
        elif ch == 6:
            video_link = str(input("Enter YouTube video link: "))
            download_youtube.download(video_link)

            video_name = download_youtube.find_mp4_and_copy_to_folder(video_link)
            combined_predict.predict_video(config.INPUT_FOLDER_PATH + video_name)
        elif ch == 7:
            video_name = str(input("Enter video name in input path: "))
            video_name = face_utils.find_filename_match(
                video_name, config.INPUT_FOLDER_PATH
            )
            print("video_path", video_name)
            combined_utils.visualise_emotions(
                video_name,
                config.OUTPUT_FOLDER_PATH
                + f"video_output_{video_name.split(os.sep)[-1].split('.')[0]}.csv",
            )
    elif ch == 4:
        if len(sys.argv) <= 2:
            ch = int(
                input(
                    "1) Tune hyperparameters\n2) Train model\n3) Preprocess data\n4) Predict emotion\n5) Predict from test set\n6) Predict from YouTube video\n7) Visualise video predictions\n\nEnter your choice: "
                )
            )
        else:
            ch = int(sys.argv[2])

        if ch == 1:
            if len(sys.argv) > 2:
                ch = int(sys.argv[3])
            else:
                ch = int(input("Continue tuning hyperparameters? (1/0): "))
            transformer_model.hyper_parameter_optimise(load_if_exists=ch)
        elif ch == 2:
            transformer_model.train_using_best_values()
        elif ch == 3:
            best_hyperparameters = face_utils.load_dict_from_json(
                config.AUDIO_BEST_HP_JSON_SAVE_PATH
            )
            print(f"Best hyperparameters, {best_hyperparameters}")

            N_FFT = best_hyperparameters["N_FFT"]
            HOP_LENGTH = best_hyperparameters["HOP_LENGTH"]
            NUM_MFCC = best_hyperparameters["NUM_MFCC"]
            transformer_preprocess.preprocess_videos(
                N_FFT=transformer_config.tune_hp_ranges["N_FFT"][0][0],
                NUM_MFCC=transformer_config.tune_hp_ranges["NUM_MFCC"][0][0],
                HOP_LENGTH=transformer_config.tune_hp_ranges["HOP_LENGTH"][0][0],
                original_videos_folder=config.ALL_EXTRACTED_AV_FOLDERS,
                output_path=config.PREPROCESSED_AV_FOLDER_PATH,
            )
        elif ch == 4:
            video_name = str(input("Enter video name in input path: "))
            video_name = face_utils.find_filename_match(
                video_name, config.INPUT_FOLDER_PATH
            )
            print("video_path", video_name)
            transformer_predict.predict_video(video_name)
        elif ch == 5:
            transformer_model.train_model(test_on_test_set=True)
        elif ch == 6:
            video_link = str(input("Enter YouTube video link: "))
            transformer_download.download(video_link)
            video_name = transformer_download.find_mp4_and_copy_to_folder(video_link)
            transformer_predict.predict_video(config.INPUT_FOLDER_PATH + video_name)
        elif ch == 7:
            video_name = str(input("Enter video name in input path: "))
            video_name = face_utils.find_filename_match(
                video_name, config.INPUT_FOLDER_PATH
            )
            print("video_path", video_name)
            transformer_utils.visualise_emotions(video_name)
