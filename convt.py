import pickle
from tensorflow.keras.models import load_model

def convert_h5_to_pickle(h5_file_path, pickle_file_path):
    
  
    model = load_model("xception_deepfake_image.h5")
    
  
    with open(pickle_file_path, 'wb') as file:
        pickle.dump(model, file)
    
    print(f"Model converted and saved to {pickle_file_path}")

# Example usage
convert_h5_to_pickle('xception_deepfake_image.h5', 'xception_deepfake_image.pkl')
