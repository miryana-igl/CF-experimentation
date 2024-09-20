import streamlit as st
import streamlit_survey as ss
from playsound import playsound
from sklearn.ensemble import RandomForestClassifier
import dice_ml
import pandas as pd
import serial
import time
from collect import *
from loader import DataLoader
from person import YourData
import plotly.express as px
import os 
from sklearn.metrics.pairwise import euclidean_distances
import csv


# Define survey
survey = ss.StreamlitSurvey("Progress Bar Example")
pages = survey.pages(5, progress_bar=True, on_submit=lambda: st.success("Thank you for participating!"))
port = "/dev/cu.usbmodem11101"
baud = 250000


# Set up the serial connection
arduino_serial = serial.Serial(port, baud)

# Load data
filepath = "data/arduino_data.csv"
arduino_data = pd.read_csv(filepath)  # Original data for predictions
user_friendly_personas = pd.read_csv('data/personas_user_friendly.csv')  # User-friendly data
rf = RandomForestClassifier()

if 'user_prediction' not in st.session_state:
    st.session_state['user_prediction'] = None
    st.session_state['sex_values'] = 50 / 7.0
    st.session_state['embarked_values'] = 50 / 7.0
    st.session_state['passenger_class_values'] = 50 / 7.0
    st.session_state['age_values'] = 50 / 7.0
    st.session_state['sibsp_values'] = 50 / 7.0
    st.session_state['parch_values'] = 50 / 7.0
    st.session_state['fare_values'] = 50 / 7.0

if 'user_prediction_post' not in st.session_state:
    st.session_state['sex_values_post'] = 50 / 7.0
    st.session_state['embarked_values_post'] = 50 / 7.0
    st.session_state['passenger_class_values_post'] = 50 / 7.0
    st.session_state['age_values_post'] = 50 / 7.0
    st.session_state['sibsp_values_post'] = 50 / 7.0
    st.session_state['parch_values_post'] = 50 / 7.0
    st.session_state['fare_values_post'] = 50 / 7.0

def wait_for_button_press():
    while True:
        if arduino_serial.in_waiting > 0:
            line = arduino_serial.readline().decode('utf-8').strip()
            if line == "START":
                return

def send_command(command):
    try:
        arduino_serial.write(command.encode())  # Send the command as bytes
        st.toast("Data collection complete!",icon="✅")
    except Exception as e:
        st.write(f"Failed to send command: {e}")  # Error handling


with pages:
    if pages.current == 0:
        col1, col2, col3 = st.columns([1, 2, 1])
        participant_id = st.text_input("Enter Participant ID:")
        st.session_state['participant_id'] = participant_id
        if st.button('Start Experiment'):
            # Wait for the button press from Arduino
        # if st.text_input("Enter your name"):
            st.write("Experiment started at: ", time.ctime())
            st.write("Please press the yellow button to start scanning data.")
            wait_for_button_press()
            st.toast("Starting data collection...")

            
            # Arduino Data Collector
            collector = ArduinoDataCollector(port, baud, filepath)
            playsound('/Users/miry/Documents/repos/CF-experimentation/audio/start.mp3')
            # Define the color map
            colour_map = {
                'Fare': '#8864ab',    # Purple
                'Age': '#df9f26',     # Yellow
                'Passenger Class': '#c74435',  # Red
                'Siblings or Spouse': '#407e3c',   # Green
                'Parents or Children': '#4b60ac',    # Blue
                'Sex': '#e4592d', # Orange
                'Embarked': '#d14282', # Pink
            }

            def update_placeholder(label, value):
                if label == "Passenger Class:":
                    col2.markdown(f"<p style='color:{colour_map['Passenger Class']};'><strong>Passenger Class:</strong> {value}</p>", unsafe_allow_html=True)
                elif label == "Sex:":
                    col2.markdown(f"<p style='color:{colour_map['Sex']};'><strong>Sex:</strong> {value}</p>", unsafe_allow_html=True)
                elif label == "Age:":
                    col2.markdown(f"<p style='color:{colour_map['Age']};'><strong>Age:</strong> {value}</p>", unsafe_allow_html=True)
                elif label == "Siblings or Spouse:":
                    col2.markdown(f"<p style='color:{colour_map['Siblings or Spouse']};'><strong>Siblings or Spouse:</strong> {value}</p>", unsafe_allow_html=True)
                elif label == "Parents or Children:":
                    col2.markdown(f"<p style='color:{colour_map['Parents or Children']};'><strong>Parents or Children:</strong> {value}</p>", unsafe_allow_html=True)
                elif label == "Fare:":
                    col2.markdown(f"<p style='color:{colour_map['Fare']};'><strong>Fare:</strong> £{value}</p>", unsafe_allow_html=True)
                elif label == "Embarked from:":
                    col2.markdown(f"<p style='color:{colour_map['Embarked']};'><strong>Embarked from:</strong> {value}</p>", unsafe_allow_html=True)
                elif label == "Survived:":
                    col2.markdown(f"<p style='color:#FFFFFFF;'><strong>Survived:</strong> {value}</p>", unsafe_allow_html=True)

            collector.collect_data(update_placeholder)
            send_command('done')
            YourData.save_user_friendly_csv()
        st.write("How much do you think each factor has an influence on the survival outcome?")
            
        # Retrieve and store each value in session state
        st.session_state['sex_values'] = st.slider("Sex:", min_value=0, max_value=100, value=int(st.session_state['sex_values'] * 7.0)) / 7.0
        st.session_state['embarked_values'] = st.slider("Embarked:", min_value=0, max_value=100, value=int(st.session_state['embarked_values'] * 7.0)) / 7.0
        st.session_state['passenger_class_values'] = st.slider("Passenger Class:", min_value=0, max_value=100, value=int(st.session_state['passenger_class_values'] * 7.0)) / 7.0
        st.session_state['age_values'] = st.slider("Age:", min_value=0, max_value=100, value=int(st.session_state['age_values'] * 7.0)) / 7.0
        st.session_state['sibsp_values'] = st.slider("Siblings / Spouse:", min_value=0, max_value=100, value=int(st.session_state['sibsp_values'] * 7.0)) / 7.0
        st.session_state['parch_values'] = st.slider("Parents / Children:", min_value=0, max_value=100, value=int(st.session_state['parch_values'] * 7.0)) / 7.0
        st.session_state['fare_values'] = st.slider("Fare:", min_value=0, max_value=100, value=int(st.session_state['fare_values'] * 7.0)) / 7.0

        if st.button("Save answers"):
                    csv_file_path = 'data/pre_values_hybrid.csv'

                    file_exists = os.path.isfile(csv_file_path)

                    with open(csv_file_path, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        
                        # Write header only if the file does not already exist
                        if not file_exists:
                            writer.writerow(["Participant ID","Sex", "Embarked", "Passenger Class", "Age", "Siblings / Spouse", "Parents / Children", "Fare"])
    
                        writer.writerow([
                            st.session_state['participant_id'],
                            st.session_state['sex_values'],
                            st.session_state['embarked_values'],
                            st.session_state['passenger_class_values'],
                            st.session_state['age_values'],
                            st.session_state['sibsp_values'],
                            st.session_state['parch_values'],
                            st.session_state['fare_values']
                        ])

                    st.success(f"Data saved to {csv_file_path}")

        total_value = st.session_state['sex_values'] + st.session_state['embarked_values'] + st.session_state['passenger_class_values'] + st.session_state['age_values'] + st.session_state['sibsp_values'] + st.session_state['parch_values'] + st.session_state['fare_values']
        total_value = 100  # Ensure the total value is set to 100


        # Column 2: Placeholders for data
        with col2:
            st.empty()  # placeholders are updated dynamically within the `update_placeholder` function

    # Column 3: Empty or additional information
    if pages.current == 1:
        # if st.button("Run the prediction  "):
            st.write("Loading data and making predictions...")
            with st.container():
                data_loader = DataLoader()
                data_loader.load_dataset()
                data_loader.preprocess_data()
                collector = ArduinoInputData()
                
                X_train, y_train = data_loader.get_data_split()
                st.session_state.X_test, st.session_state.y_test, st.session_state.input_data = collector.get_input()
                
                rf = RandomForestClassifier()
                rf.fit(X_train, y_train)
                st.session_state.y_pred = rf.predict(st.session_state.X_test)
                y_pred_proba = rf.predict_proba(st.session_state.X_test)
                
                # Convert probabilities to percentages
                survived_probability = y_pred_proba[0][1]
                did_not_survive_probability = y_pred_proba[0][0]
                survived_percentage = survived_probability * 100
                did_not_survive_percentage = did_not_survive_probability * 100

                # Define the sizes of the rectangles based on percentages
                survived_width = survived_percentage
                did_not_survive_width = did_not_survive_percentage

                YourData.save_user_friendly_csv()
                # Define the HTML and CSS for displaying rectangles
                html = f"""
                <div style="display: flex; justify-content: center; align-items: center; margin-top: 20px;">
                    <div style="width: {survived_width}%; height: 100px; background-color: #69e461; color: black; text-align: center; line-height: 100px; font-size: 20px; margin-right: 10px;">
                        {survived_percentage:.0f}%
                    </div>
                    <div style="width: {did_not_survive_width}%; height: 100px; background-color: #ff2c2c; color: black; text-align: center; line-height: 100px; font-size: 20px;">
                        {did_not_survive_percentage:.0f}%
                    </div>
                </div>
                """
                st.markdown(html, unsafe_allow_html=True)

                ####### DICE COUNTERFACTUALS #############
                # Counterfactual explanations
                data_dice = dice_ml.Data(dataframe=st.session_state.input_data,
                                        continuous_features=["Age", "Fare", "Pclass", 'SibSp', 'Parch'],
                                        outcome_name='Survived')

                rf_dice = dice_ml.Model(model=rf, backend="sklearn")
                explainer = dice_ml.Dice(data_dice, rf_dice, method="random")

                input_datapoint = st.session_state.X_test[0:1]
                features_to_vary = ['Age', 'Fare', 'Pclass', 'SibSp', 'Parch']
                permitted_range = {'Age': [0, 80], 'Fare': [1, 10000], 'Pclass': [1, 3],'SibSp':[0,6],'Parch':[0,6]}
                
                cf = explainer.generate_counterfactuals(input_datapoint,
                                                        total_CFs=3,
                                                        desired_class="opposite",
                                                        permitted_range=permitted_range,
                                                        features_to_vary=features_to_vary,
                                                        proximity_weight=1.5,
                                                        diversity_weight=1.0,
                                                        verbose=False)
                
                st.session_state.cf_df = cf.cf_examples_list[0].final_cfs_df

            def generate_textual_explanation(original_instance, counterfactual_instance, prediction_column='Survived'):
                changes = []
                for column in original_instance.index:
                    if original_instance[column] != counterfactual_instance[column]:
                        original_value = original_instance[column]
                        new_value = counterfactual_instance[column]
                        if column == 'Fare':
                            original_value = f"£{original_value}"
                            new_value = f"£{new_value}"
                        elif column == 'SibSp':
                            original_value = f"{original_value} Sibling/Spouse" if original_value in [0, 1] else f"{original_value} Siblings"
                        elif column == 'Parch':
                            if original_value == 0:
                                original_value = "No Parent/Child"
                            elif original_value == 1:
                                original_value = "1 Parent/Child"
                            elif original_value == 2:
                                original_value = "2 Parents/Children"
                            else:
                                original_value = f"{original_value} Children"
                            new_value = f"{new_value} Children" if new_value > 1 else "Parent/Child" if new_value == 1 else "No Parent/Child"
                        elif column == 'Pclass':
                            original_value = f"{original_value}st Passenger Class" if original_value == 1 else f"{original_value}nd Passenger Class" if original_value == 2 else f"{original_value}rd Passenger Class"
                            new_value = f"{new_value}st" if new_value == 1 else f"{new_value}nd" if new_value == 2 else f"{new_value}rd"

                        changes.append((column, original_value, new_value))
                
                change_descriptions = []
                for feature, original_value, new_value in changes:
                    if feature in ['SibSp', 'Parch', 'Pclass']:
                        change_descriptions.append(f"from **{original_value}** to **{new_value}**")
                    else:
                        change_descriptions.append(f"the **{feature}** from **{original_value}** to **{new_value}**")
                
                changes_text = " and ".join(change_descriptions)
                new_prediction = counterfactual_instance[prediction_column]
                new_prediction_label = ':green[SURVIVED]' if new_prediction == 1 else ':red[DID NOT SURVIVE]'
                
                return changes_text, new_prediction_label

            if st.session_state.input_data is not None and st.session_state.cf_df is not None:
                # Ensure X_test is a DataFrame and handle the first row correctly
                if isinstance(st.session_state.X_test, pd.DataFrame):
                    original_instance = st.session_state.X_test.iloc[0]
                else:
                    st.write("Error: X_test is not a DataFrame.")
                    st.stop()

                original_prediction = st.session_state.y_test.iloc[0]
                original_prediction_label = ':green[SURVIVED]' if original_prediction == 1 else ':red[DID NOT SURVIVE]'
                ai_prediction = st.session_state.y_pred[0]
                ai_prediction_label = ':green[SURVIVED]' if ai_prediction == 1 else ':red[DID NOT SURVIVE]'
                result = ':red[INCORRECT]' if original_prediction != ai_prediction else ':green[CORRECT]'


                textual_explanations = []
                new_prediction_label = None
                for i in range(st.session_state.cf_df.shape[0]):
                    counterfactual_instance = st.session_state.cf_df.iloc[i]
                    changes_text, new_prediction_label = generate_textual_explanation(original_instance, counterfactual_instance)
                    textual_explanations.append(f"Changing {changes_text} will result in the prediction changing to '{new_prediction_label}'.")

                #########################################    
                correct = "/Users/miry/Documents/repos/CF-experimentation/audio/correct.mp3"
                incorrect = "/Users/miry/Documents/repos/CF-experimentation/audio/incorrect.mp3"

                st.text("Survived: 🟩   Did Not Survive: 🟥")
                st.write(f"\n")
                st.write(f"You predicted that the person {original_prediction_label} the Titanic")
                st.write(f"The AI predicted that the person  {ai_prediction_label} the Titanic")
                if original_prediction == ai_prediction:
                    
                    st.write('Your predictioin was :green[CORRECT]!')
                    playsound(correct)
                    
                else:
                    st.write('Your predictioin was :red[INCORRECT]!')
                    playsound(incorrect)

                ################# COUNTERFACTUAL CONTAINER #################

    if pages.current == 2:
        filepath = "/Users/miry/Documents/repos/CF-experimentation/data/arduino_data.csv"

        st.write("Editing Data:")
        st.write("Here you can edit the data you collected to make more precise predictions.")        
        # Check if the CSV file exists
        if os.path.exists(filepath):
            df1 = pd.read_csv(filepath)
        else:
            st.error("CSV file not found!")
            # return
        if st.button('Collect new data'):
            col1, col2, col3 = st.columns([1, 2, 1])
            st.toast("Starting data collection...")

            
            # Arduino Data Collector
            collector = ArduinoDataCollector(port, baud, filepath)
            playsound('/Users/miry/Documents/repos/CF-experimentation/audio/start.mp3')
            # Define the color map
            colour_map = {
                'Fare': '#8864ab',    # Purple
                'Age': '#df9f26',     # Yellow
                'Passenger Class': '#c74435',  # Red
                'Siblings or Spouse': '#407e3c',   # Green
                'Parents or Children': '#4b60ac',    # Blue
                'Sex': '#e4592d', # Orange
                'Embarked': '#d14282', # Pink
            }

            def update_placeholder(label, value):
                if label == "Passenger Class:":
                    col2.markdown(f"<p style='color:{colour_map['Passenger Class']};'><strong>Passenger Class:</strong> {value}</p>", unsafe_allow_html=True)
                elif label == "Sex:":
                    col2.markdown(f"<p style='color:{colour_map['Sex']};'><strong>Sex:</strong> {value}</p>", unsafe_allow_html=True)
                elif label == "Age:":
                    col2.markdown(f"<p style='color:{colour_map['Age']};'><strong>Age:</strong> {value}</p>", unsafe_allow_html=True)
                elif label == "Siblings or Spouse:":
                    col2.markdown(f"<p style='color:{colour_map['Siblings or Spouse']};'><strong>Siblings or Spouse:</strong> {value}</p>", unsafe_allow_html=True)
                elif label == "Parents or Children:":
                    col2.markdown(f"<p style='color:{colour_map['Parents or Children']};'><strong>Parents or Children:</strong> {value}</p>", unsafe_allow_html=True)
                elif label == "Fare:":
                    col2.markdown(f"<p style='color:{colour_map['Fare']};'><strong>Fare:</strong> £{value}</p>", unsafe_allow_html=True)
                elif label == "Embarked from:":
                    col2.markdown(f"<p style='color:{colour_map['Embarked']};'><strong>Embarked from:</strong> {value}</p>", unsafe_allow_html=True)
                elif label == "Survived:":
                    col2.markdown(f"<p style='color:#FFFFFFF;'><strong>Survived:</strong> {value}</p>", unsafe_allow_html=True)

            collector.collect_data(update_placeholder)
            send_command('done')
            YourData.save_user_friendly_csv()

#################### START OF FORM ####################
        with st.form(key='edit_form'):
            st.write("Original Data")
            YourData.save_user_friendly_csv()
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input('Age', min_value=0, max_value=80, value=int(df1['Age'][0]), key='Age')
            with col2:
                fare = st.number_input('Fare', min_value=0.00, max_value=10000.00, value=float(df1['Fare'][0]), key='Fare')

            submit_button = st.form_submit_button(label='Re-run prediction')

            if submit_button:
                df1.loc[0, 'Age'] = age
                df1.loc[0, 'Fare'] = fare

                # Save the updated DataFrame back to the CSV file
                df1.to_csv(filepath, index=False)
                st.success('Data updated successfully!')
#################### END OF FORM ####################
#################### RUN PREDICTION ON UPDATED DATA ####################

                with st.container():
                    data_loader = DataLoader()
                    data_loader.load_dataset()
                    data_loader.preprocess_data()
                    collector = ArduinoInputData()
                    
                    X_train, y_train = data_loader.get_data_split()
                    st.session_state.X_test, st.session_state.y_test, st.session_state.input_data = collector.get_input()
                    
                    rf = RandomForestClassifier()
                    rf.fit(X_train, y_train)
                    st.session_state.y_pred = rf.predict(st.session_state.X_test)
                    y_pred_proba = rf.predict_proba(st.session_state.X_test)
                    
                    # Convert probabilities to percentages
                    survived_probability = y_pred_proba[0][1]
                    did_not_survive_probability = y_pred_proba[0][0]
                    survived_percentage = survived_probability * 100
                    did_not_survive_percentage = did_not_survive_probability * 100

                    # Define the sizes of the rectangles based on percentages
                    survived_width = survived_percentage
                    did_not_survive_width = did_not_survive_percentage

                    YourData.save_user_friendly_csv()
                    # Define the HTML and CSS for displaying rectangles
                    html = f"""
                    <div style="display: flex; justify-content: center; align-items: center; margin-top: 20px;">
                        <div style="width: {survived_width}%; height: 100px; background-color: #69e461; color: black; text-align: center; line-height: 100px; font-size: 20px; margin-right: 10px;">
                            {survived_percentage:.0f}%
                        </div>
                        <div style="width: {did_not_survive_width}%; height: 100px; background-color: #ff2c2c; color: black; text-align: center; line-height: 100px; font-size: 20px;">
                            {did_not_survive_percentage:.0f}%
                        </div>
                    </div>
                    """
                    st.markdown(html, unsafe_allow_html=True)

                    ####### DICE COUNTERFACTUALS #############
                    # Counterfactual explanations
                    data_dice = dice_ml.Data(dataframe=st.session_state.input_data,
                                            continuous_features=["Age", "Fare", "Pclass", 'SibSp', 'Parch'],
                                            outcome_name='Survived')

                    rf_dice = dice_ml.Model(model=rf, backend="sklearn")
                    explainer = dice_ml.Dice(data_dice, rf_dice, method="random")

                    input_datapoint = st.session_state.X_test[0:1]
                    features_to_vary = ['Age', 'Fare', 'Pclass', 'SibSp', 'Parch']
                    permitted_range = {'Age': [0, 80], 'Fare': [1, 10000], 'Pclass': [1, 3],'SibSp':[0,6],'Parch':[0,6]}
                    
                    cf = explainer.generate_counterfactuals(input_datapoint,
                                                            total_CFs=3,
                                                            desired_class="opposite",
                                                            permitted_range=permitted_range,
                                                            features_to_vary=features_to_vary,
                                                            proximity_weight=1.5,
                                                            diversity_weight=1.0,
                                                            verbose=False)
                    
                    st.session_state.cf_df = cf.cf_examples_list[0].final_cfs_df

                def generate_textual_explanation(original_instance, counterfactual_instance, prediction_column='Survived'):
                    changes = []
                    for column in original_instance.index:
                        if original_instance[column] != counterfactual_instance[column]:
                            original_value = original_instance[column]
                            new_value = counterfactual_instance[column]
                            if column == 'Fare':
                                original_value = f"£{original_value}"
                                new_value = f"£{new_value}"
                            elif column == 'SibSp':
                                original_value = f"{original_value} Sibling/Spouse" if original_value in [0, 1] else f"{original_value} Siblings"
                            elif column == 'Parch':
                                if original_value == 0:
                                    original_value = "No Parent/Child"
                                elif original_value == 1:
                                    original_value = "1 Parent/Child"
                                elif original_value == 2:
                                    original_value = "2 Parents/Children"
                                else:
                                    original_value = f"{original_value} Children"
                                new_value = f"{new_value} Children" if new_value > 1 else "Parent/Child" if new_value == 1 else "No Parent/Child"
                            elif column == 'Pclass':
                                original_value = f"{original_value}st Passenger Class" if original_value == 1 else f"{original_value}nd Passenger Class" if original_value == 2 else f"{original_value}rd Passenger Class"
                                new_value = f"{new_value}st" if new_value == 1 else f"{new_value}nd" if new_value == 2 else f"{new_value}rd"

                            changes.append((column, original_value, new_value))
                    
                    change_descriptions = []
                    for feature, original_value, new_value in changes:
                        if feature in ['SibSp', 'Parch', 'Pclass']:
                            change_descriptions.append(f"from **{original_value}** to **{new_value}**")
                        else:
                            change_descriptions.append(f"the **{feature}** from **{original_value}** to **{new_value}**")
                    
                    changes_text = " and ".join(change_descriptions)
                    new_prediction = counterfactual_instance[prediction_column]
                    new_prediction_label = ':green[SURVIVED]' if new_prediction == 1 else ':red[DID NOT SURVIVE]'
                    
                    return changes_text, new_prediction_label

                if st.session_state.input_data is not None and st.session_state.cf_df is not None:
                    # Ensure X_test is a DataFrame and handle the first row correctly
                    if isinstance(st.session_state.X_test, pd.DataFrame):
                        original_instance = st.session_state.X_test.iloc[0]
                    else:
                        st.write("Error: X_test is not a DataFrame.")
                        st.stop()

                    original_prediction = st.session_state.y_test.iloc[0]
                    original_prediction_label = ':green[SURVIVED]' if original_prediction == 1 else ':red[DID NOT SURVIVE]'
                    ai_prediction = st.session_state.y_pred[0]
                    ai_prediction_label = ':green[SURVIVED]' if ai_prediction == 1 else ':red[DID NOT SURVIVE]'
                    result = ':red[INCORRECT]' if original_prediction != ai_prediction else ':green[CORRECT]'


                    textual_explanations = []
                    new_prediction_label = None
                    for i in range(st.session_state.cf_df.shape[0]):
                        counterfactual_instance = st.session_state.cf_df.iloc[i]
                        changes_text, new_prediction_label = generate_textual_explanation(original_instance, counterfactual_instance)
                        textual_explanations.append(f"Changing {changes_text} will result in the prediction changing to '{new_prediction_label}'.")

                    #########################################    
                    correct = "/Users/miry/Documents/repos/CF-experimentation/audio/correct.mp3"
                    incorrect = "/Users/miry/Documents/repos/CF-experimentation/audio/incorrect.mp3"

                    st.text("Survived: 🟩   Did Not Survive: 🟥")
                    st.write(f"\n")
                    st.write(f"You predicted that the person {original_prediction_label} the Titanic")
                    st.write(f"The AI predicted that the person  {ai_prediction_label} the Titanic")
                    if original_prediction == ai_prediction:
                        
                        st.write('Your predictioin was :green[CORRECT]!')
                        playsound(correct)
                        
                    else:
                        st.write('Your predictioin was :red[INCORRECT]!')
                        playsound(incorrect)

                    ################# COUNTERFACTUAL CONTAINER #################


                    if new_prediction_label:
                        st.write("\n")
                        
                    # Create three columns for the scenarios
                    cols = st.columns(3)
                    for i in range(3):
                        with cols[i]:
                            # Create a container within each column
                            with st.container(height=300):
                                st.subheader(f"Scenario {i+1}:")
                                st.markdown(f"{textual_explanations[i]}")
                else:
                    st.write("No data available to generate textual explanations.")

#################### END OF RUN PREDICTION ON UPDATED DATA ####################
    elif pages.current == 3:
                ########## FINAL PAGE ##########
                st.write("Page 4")
                with st.form(key='edit_form'):
                    st.write("How much do you think each factor has an influence on the survival outcome?")
                        
                    # Retrieve and store each value in session state
                    st.session_state['sex_values_post'] = st.slider("Sex:", min_value=0, max_value=100, value=int(st.session_state['sex_values'] * 7.0)) / 7.0
                    st.session_state['embarked_values_post'] = st.slider("Embarked:", min_value=0, max_value=100, value=int(st.session_state['embarked_values'] * 7.0)) / 7.0
                    st.session_state['passenger_class_values_post'] = st.slider("Passenger Class:", min_value=0, max_value=100, value=int(st.session_state['passenger_class_values'] * 7.0)) / 7.0
                    st.session_state['age_values_post'] = st.slider("Age:", min_value=0, max_value=100, value=int(st.session_state['age_values'] * 7.0)) / 7.0
                    st.session_state['sibsp_values_post'] = st.slider("Siblings / Spouse:", min_value=0, max_value=100, value=int(st.session_state['sibsp_values'] * 7.0)) / 7.0
                    st.session_state['parch_values_post'] = st.slider("Parents / Children:", min_value=0, max_value=100, value=int(st.session_state['parch_values'] * 7.0)) / 7.0
                    st.session_state['fare_values_post'] = st.slider("Fare:", min_value=0, max_value=100, value=int(st.session_state['fare_values'] * 7.0)) / 7.0

                    submit_button = st.form_submit_button(label='Save your answers')
                    if submit_button:
                        # Define CSV file path
                        csv_file_path = 'data/post_values_hybrid.csv'
                        # Check if the file already exists
                        file_exists = os.path.isfile(csv_file_path)

                        with open(csv_file_path, mode='a', newline='') as file:
                            writer = csv.writer(file)
                            
                            # Write header only if the file does not already exist
                            if not file_exists:
                                writer.writerow(["Participant ID","Sex", "Embarked", "Passenger Class", "Age", "Siblings / Spouse", "Parents / Children", "Fare"])
                            
                            writer.writerow([
                                st.session_state['participant_id'],
                                st.session_state['sex_values_post'],
                                st.session_state['embarked_values_post'],
                                st.session_state['passenger_class_values_post'],
                                st.session_state['age_values_post'],
                                st.session_state['sibsp_values_post'],
                                st.session_state['parch_values_post'],
                                st.session_state['fare_values_post']
                            ])

                            st.success(f"Data saved to {csv_file_path}")
                    total_value = st.session_state['sex_values_post'] + st.session_state['embarked_values_post'] + st.session_state['passenger_class_values_post'] + st.session_state['age_values_post'] + st.session_state['sibsp_values_post'] + st.session_state['parch_values_post'] + st.session_state['fare_values_post']
                    total_value = 100  # Ensure the total value is set to 100
                

    elif pages.current == 4:
        

            # Set the title of the Streamlit app
            st.title("Hybrid UI User Experience Questionnaire")

            # Embed the Google Form using the iframe URL
            st.components.v1.iframe(
                src="https://docs.google.com/forms/d/e/1FAIpQLSeshEUi0DRuEVMUQ6KQ2S9DuNb9eVOwXSsRgeSubbO1kAHlUQ/viewform?embedded=true", 
                width=640, 
                height=1260, 
                scrolling=True
            )
