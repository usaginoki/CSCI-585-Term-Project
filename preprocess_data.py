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
