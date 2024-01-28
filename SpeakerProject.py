import os
import numpy as np
import librosa


def load_audio_file(file_path):
    # Load an audio file and return the audio signal and sample rate.
    audio_signal, sample_rate = librosa.load(file_path)
    return audio_signal, sample_rate


def calculate_dominant_frequency(audio_signal, sample_rate):
    # Calculate the dominant frequency from an audio signal using FFT.
    fft_result = np.fft.fft(audio_signal)
    magnitude_spectrum = np.abs(fft_result)

    frequency_axis = np.fft.fftfreq(len(audio_signal), 1/sample_rate)
    positive_frequencies = frequency_axis[:len(audio_signal)//2]
    positive_magnitudes = magnitude_spectrum[:len(audio_signal)//2]

    dominant_freq_index = np.argmax(positive_magnitudes)
    dominant_frequency = round(positive_frequencies[dominant_freq_index], 4)

    return dominant_frequency


def calculate_average_feature_vector(dominant_freq_list):
    # Calculate the average feature vector from a list of dominant frequencies.
    return round(np.average(dominant_freq_list[:10]), 4)


def calculate_recognition_accuracy(test_feature_vectors, closest_match_values):
    # Calculate recognition accuracy based on the number of correct recognitions.
    correct_recognitions = sum(val in test_feature_vectors for val in closest_match_values)
    total_recognitions = len(closest_match_values)

    recognition_accuracy = (correct_recognitions / total_recognitions) * 100 if total_recognitions != 0 else 0
    return round(recognition_accuracy, 4)


def main():
    # Define the folder path containing audio files
    AUDIO_FOLDER_PATH = r'C:\Users\Hussein Essam\OneDrive\Desktop\CCEC Junior\Signals Project\Data\All'

    # Dictionaries to store dominant frequencies, feature vectors, and test vectors
    dominant_freq_dict = {"SpeakerA": [], "SpeakerB": [], "SpeakerC": []}
    feature_vector_dict = {"SpeakerA": 0, "SpeakerB": 0, "SpeakerC": 0}
    test_feature_vector_dict = {"SpeakerA": [], "SpeakerB": [], "SpeakerC": []}
    test_vector_list = []

    # Process each speaker
    for speaker in dominant_freq_dict.keys():
        # Process each audio file for the speaker
        for i in range(1, 16):
            file_name = f"{speaker}_{i}.wav"
            file_path = os.path.join(AUDIO_FOLDER_PATH, file_name)

            # Load audio file
            audio_signal, sample_rate = load_audio_file(file_path)

            # Calculate dominant frequency
            dominant_frequency = calculate_dominant_frequency(audio_signal, sample_rate)
            dominant_freq_dict[speaker].append(dominant_frequency)

        # Calculate average feature vector and store test vectors
        feature_vector_dict[speaker] = calculate_average_feature_vector(dominant_freq_dict[speaker][:10])
        test_feature_vector_dict[speaker] = dominant_freq_dict[speaker][10:]
        test_vector_list.extend(dominant_freq_dict[speaker][10:])

    # List of average feature vectors for all speakers
    average_feature_vector_list = [feature_vector_dict[val] for val in feature_vector_dict]

    # Dictionary to store closest matching feature vectors for each speaker
    closest_match_dict = {"SpeakerA": [], "SpeakerB": [], "SpeakerC": []}

    # Process each test vector
    for index, test_val in enumerate(test_vector_list):
        # Find the closest matching feature vector
        closest_value = min(average_feature_vector_list, key=lambda x: abs(x - test_val))
        closest_index = average_feature_vector_list.index(closest_value)
        recognized_student = list(feature_vector_dict.keys())[closest_index]
        print(f"For test {index + 1}, the recognized student was {recognized_student}")
        closest_match_dict[recognized_student].append(test_val)

    # Dictionary to store recognition accuracy for each speaker
    recognition_accuracy_dict = {"SpeakerA": 0, "SpeakerB": 0, "SpeakerC": 0}

    # Calculate recognition accuracy for each speaker
    for speaker in recognition_accuracy_dict:
        recognition_accuracy = calculate_recognition_accuracy(
            test_feature_vector_dict[speaker], closest_match_dict[speaker]
        )
        recognition_accuracy_dict[speaker] = recognition_accuracy
        print(f"Recognition accuracy for {speaker}: {recognition_accuracy}%")

    # Calculate overall recognition accuracy for the program
    overall_accuracy = round(np.average(list(recognition_accuracy_dict.values())), 4)
    print(f"Overall recognition accuracy: {overall_accuracy}%")


if __name__ == "__main__":
    main()
