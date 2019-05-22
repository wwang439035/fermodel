TRAIN_DATA_DIR = 'gs://fer-project-241205-vcm/'
VALIDATION_DATA_DIR = 'data_files/validation/'

TRAINED_MODEL_PATH = 'data_files/model/fermodel.pkl'


BATCH_SIZE = 128
IMAGE_SIZE = 48


EMOTIONS_LIST = ["Angry", "Happy", "Neutral", "Sad", "Surprise", "Disgust"]
NUM_OF_CLASS = len(EMOTIONS_LIST)
EPOCHS = 20

IS_ANALYSIS = False

GCP_BUCKET_URL = 'gs://fer-project-mlengine/'
TRAIN_DATA_CSV_FILE = 'data_files/train_data.csv'
VALIDATION_DATA_CSV_FILE = 'data_files/validation_data.csv'
