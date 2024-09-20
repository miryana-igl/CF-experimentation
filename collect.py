import serial
import pandas as pd
from playsound import playsound

class ArduinoDataCollector:

    def __init__(self, port, baud_rate, file_path):
        self.arduinoData = serial.Serial(port, baud_rate)
        self.file_path = file_path

        self.samples = 8
        self.print_labels = False

        # Variables to store collected values
        self.Pclass = self.Sex = self.Age = self.SibSp = self.Parch = self.Fare = self.Embarked = self.Survived = None

# Lists of NFC UIDs
        self.pclass_uids = ["04 9E F9 55 6F 61 80", "04 50 22 4E 6F 61 80", "04 16 AC 55 6F 61 80"]
        self.sex_uids = ["04 33 25 51 6F 61 80", "04 80 F2 6D 8F 61 80"]
        self.age_uids = ["04 61 7F 53 6F 61 80", "04 25 C4 4E 6F 61 81", "04 8C 3C 53 6F 61 80", 
                        "04 87 78 56 6F 61 80", "04 BA 63 56 6F 61 80", "04 A7 9C 4E 6F 61 80", 
                        "04 49 05 2F 4F 61 80"]
        self.sibsp_uids = ["04 30 C1 55 6F 61 80", "04 BA 01 4F 6F 61 80", "04 25 48 56 6F 61 81", 
                        "04 B6 7E 56 6F 61 80", "04 5B 76 4E 6F 61 80", "04 1B 35 56 6F 61 80", 
                        "04 33 A2 2E 4F 61 80"]
        self.parch_uids = ["04 86 0F 53 6F 61 80", "04 CD 9A 54 6F 61 80", "04 CB E6 4E 6F 61 80", 
                        "04 2B AD 65 5F 61 80", "04 31 11 63 5F 61 80", "04 64 51 66 5F 61 80", 
                        "04 B7 D2 30 4F 61 81"]
        self.fare_uids = ["04 43 B4 6E 8F 61 80", "04 05 3B 51 6F 61 80", "04 E6 A7 22 CE 11 90", 
                        "04 A0 4D 6E 8F 61 80", "04 6F C9 22 CE 11 90", "04 7D C5 22 CE 11 91"]
        self.embarked_uids = ["04 AD 01 22 CE 11 94", "04 2E 64 22 CE 11 91", "04 3F 47 22 CE 11 90"]
        self.survived_uids = ["04 01 64 22 CE 11 91", "04 B7 BA 22 CE 11 90"]

        self.embarked_s = ["England"]
        self.embarked_c = ["France"]
        self.embarked_q = ["Ireland"]
        self.sex_female = ["Female"]
        self.sex_male = ["Male"]
        self.survived_yes = ["yes"]

        self.create_file()
    def start(self, update_placeholder):
        # Wait for the button press to start data collection
        self.start_collection.wait_for_button_press()
        self.collect_data(update_placeholder)

    def create_file(self):
        try:
            with open(self.file_path, "w") as file:
                file.write("Sex_female,Sex_male,Embarked_C,Embarked_Q,Embarked_S,Pclass,Age,SibSp,Parch,Fare,Survived\n")
            print("Created file")
        except IOError as e:
            print(f"Error creating file: {e}")

    def check_variables(self):
        return all([self.Pclass, self.Sex, self.Age, self.SibSp, self.Parch, self.Fare, self.Embarked, self.Survived])

    def collect_data(self, update_placeholder):
        ding = '/Users/miry/Documents/repos/CF-experimentation/audio/ding.mp3'
        dong = '/Users/miry/Documents/repos/CF-experimentation/audio/complete.mp3'
        try:
            with open(self.file_path, "a") as file:
                while True:
                    if self.arduinoData.in_waiting > 0:
                        data = self.arduinoData.readline().strip().decode('utf-8')

                        split_data = data.split(',', 1)
                        if len(split_data) == 2:
                            uid, value = split_data
                            value = value.strip()

                            if uid in self.pclass_uids:
                                self.Pclass = value
                                update_placeholder("Passenger Class:", self.Pclass)
                                playsound(ding)
                            elif uid in self.sex_uids:
                                self.Sex = value
                                update_placeholder("Sex:", self.Sex)
                                playsound(ding)
                            elif uid in self.age_uids:
                                self.Age = value
                                update_placeholder("Age:", self.Age)
                                playsound(ding)
                            elif uid in self.sibsp_uids:
                                self.SibSp = value
                                update_placeholder("Siblings or Spouse:", self.SibSp)
                                playsound(ding)
                            elif uid in self.parch_uids:
                                self.Parch = value
                                update_placeholder("Parents or Children:", self.Parch)
                                playsound(ding)
                            elif uid in self.fare_uids:
                                self.Fare = value
                                update_placeholder("Fare:", self.Fare)
                                playsound(ding)
                            elif uid in self.embarked_uids:
                                self.Embarked = value
                                update_placeholder("Embarked from:", self.Embarked)
                                playsound(ding)
                            elif uid in self.survived_uids:
                                self.Survived = value
                                update_placeholder("Survived:", self.Survived)
                                playsound(ding)

                            if self.check_variables():
                                Sex_female = 1 if self.Sex in self.sex_female else 0
                                Sex_male = 1 if self.Sex in self.sex_male else 0
                                Embarked_C = 1 if self.Embarked in self.embarked_c else 0
                                Embarked_Q = 1 if self.Embarked in self.embarked_q else 0
                                Embarked_S = 1 if self.Embarked in self.embarked_s else 0
                                Survived = 1 if self.Survived in self.survived_yes else 0

                                file.write(f"{Sex_female},{Sex_male},{Embarked_C},{Embarked_Q},{Embarked_S},{self.Pclass},{self.Age},{self.SibSp},{self.Parch},{self.Fare},{Survived}\n")

                                # Reset variables for the next item
                                self.Pclass = self.Sex = self.Age = self.SibSp = self.Parch = self.Fare = self.Embarked = self.Survived = None
                                break
                        else:
                            print("Invalid data format:", data)
        except (IOError, serial.SerialException) as e:
            print(f"Error during data collection: {e}")

        print("Data collection complete!")
        playsound(dong)
class ArduinoInputData:
    def get_input(self):
        self.input_data = pd.read_csv("data/arduino_data.csv")
        self.y_body = self.input_data.iloc[:, :-1]   
        self.y_label = self.input_data.iloc[:, -1]
        return self.y_body, self.y_label, self.input_data