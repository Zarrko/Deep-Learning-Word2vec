import os
import glob
from appointment import is_appointment

# /home/zack/Desktop/ML/kenya-hansards/data/gazette-txt-files
# /home/zack/data/gazette-txt-files
path_to_folder = "/home/zack/Desktop/ML/kenya-hansards/data/gazette-txt-files"


def get_file_text(path_to_folder):
    notices = []
    for directory in os.listdir(path_to_folder):
        file_list = glob.glob(os.path.join(path_to_folder+"/"+directory, '*.txt'))

        for file in file_list:
            file_text = open(file, 'r').read()
            notices.append(file_text)

    return notices

all_notices = get_file_text(path_to_folder)
appointments = []

for notice in all_notices:
    if is_appointment(notice):
        appointments.append(notice)

print(len(appointments))


def split_appointments(appointments_list):
    half = len(appointments_list)//2
    return appointments_list[:half], appointments_list[half:]

print(appointments[0])


