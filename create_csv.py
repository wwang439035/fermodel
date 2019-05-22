from constants.constants_gcp import GCP_BUCKET_URL, TRAIN_DATA_DIR, TRAIN_DATA_CSV_FILE, VALIDATION_DATA_DIR, VALIDATION_DATA_CSV_FILE
import csv
import os


def create_csv_file(file_path, data_dir, gcp_sub_dir):
    with open(file_path, 'w') as csv_file:
        spam_writer = csv.writer(csv_file)
        expressions = next(os.walk(data_dir))[1]
        for expression in expressions:
            image_files = os.listdir(data_dir + expression)
            for image_file in image_files:
                spam_writer.writerow([GCP_BUCKET_URL + gcp_sub_dir + expression + '/' + image_file, expression])


create_csv_file(TRAIN_DATA_CSV_FILE, TRAIN_DATA_DIR, 'train_data/')
create_csv_file(VALIDATION_DATA_CSV_FILE, VALIDATION_DATA_DIR, 'validation_data/')
