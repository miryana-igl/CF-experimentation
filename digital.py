import os
import pandas as pd
import streamlit as st
import streamlit_survey as ss
from loader import DataLoader
from sklearn.ensemble import RandomForestClassifier
import dice_ml
from digi_person import YourData
import csv

# Define survey
survey = ss.StreamlitSurvey("Progress Bar Example")
pages = survey.pages(5, progress_bar=True, on_submit=lambda: st.success("Thank you for participating!"))

# Load data
personas = pd.read_csv('data/personas.csv')  # Original data for predictions
user_friendly_personas = pd.read_csv('data/personas_user_friendly.csv')  # User-friendly data
rf = RandomForestClassifier()
# Store data in session state
if 'participant_id' not in st.session_state:
    st.session_state['participant_id'] = None

if 'counter' not in st.session_state:
    st.session_state.counter = 0

if 'personas' not in st.session_state:
    st.session_state['personas'] = personas

if 'user_friendly_personas' not in st.session_state:
    st.session_state['user_friendly_personas'] = user_friendly_personas

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


if 'factors' not in st.session_state:
    st.session_state['factors'] = None

if 'selected_id' not in st.session_state:
    st.session_state['selected_id'] = None

if 'selected_label' not in st.session_state:
    st.session_state['selected_label'] = None 

if 'selected_person_display' not in st.session_state:
    st.session_state['selected_person_display'] = None

if 'survive' not in st.session_state:
    st.session_state['survive'] = None

if 'cf_df' not in st.session_state:
    st.session_state.cf_df = None

if 'textual_explanations' not in st.session_state:
    st.session_state.textual_explanations = None

if 'new_prediction_label' not in st.session_state:
    st.session_state.new_prediction_label = None

# Retrieve user-friendly personas
user_friendly_personas = st.session_state['user_friendly_personas']
personas = st.session_state['personas']

# Define custom descriptive labels for each person
custom_labels = {
    1:"Please select",
    2: "A young boy",
    3: "A young girl",
    4: "A middle-aged man",
    5: "A middle-aged woman",
    6: "An elderly man",
    7: "An elderly woman"
}

with pages:
    if pages.current == 0:
        st.write("Welcome.")
        # List of labels based on IDs in user-friendly data
        labels = [custom_labels.get(id, f"Person {id}") for id in user_friendly_personas['id']]
        # Collect Participant ID
        participant_id = st.text_input("Enter Participant ID:")
        st.session_state['participant_id'] = participant_id

        selected_label = st.selectbox("Use the drop down menu to select a person and predict their survival.", options=labels)
        if selected_label == "Please select":
            st.session_state['selected_id'] = None
            st.session_state['selected_label'] = None
            st.session_state['selected_person_display'] = None
        else:
            selected_id = [id for id, label in custom_labels.items() if label == selected_label][0]
            st.session_state['selected_id'] = selected_id
            st.session_state['selected_label'] = selected_label

            selected_person = user_friendly_personas[user_friendly_personas['id'] == selected_id]

            # Drop 'id' column from DataFrame before displaying
            selected_person_display = selected_person.drop(columns=['id'])
            st.session_state['selected_person_display'] = selected_person_display
            # Display the selected person's user-friendly details
            st.write("Carefully review the details of the selected person and predict whether they would survive the Titanic disaster.")
            st.write("**Sex** is represented in the first column")
            st.write("**Embarked** is represented in the second column, it signifies where they boarded the ship from.")
            st.write("**Passenger Class** is represented in the third column, it signifies the ticket class type of the passenger. 1st class is the highest class and 3rd the lowest.")
            st.write("**Age** is represented in the fourth column.")
            st.write("**Siblings / Spouse** is represented in the fifth column, it signifies the number of siblings or a spouse the selected person was travelling with aboard the Titanic.")
            st.write("**Parents / Children** is represented in the sixth column, it signifies the number of parents or children the selected person was travelling with aboard the Titanic.")
            st.write("**Fare** is represented in the seventh column, it signifies the amount of money paid for the ticket accounting for today's inflation.")
            st.write(f"Details for **{selected_label}**:")
            st.dataframe(selected_person_display, hide_index=True)

        if selected_label == "Please select":
            st.write("Please select a person to begin.")
        else:
            # Radio button for user to guess survival
            col1, col2 = st.columns(2)
            with col1:
                survive = st.radio("Do you think this person would survive?", ("Please Select", "Yes", "No"))
                st.session_state['survive'] = survive

            with col2:
                if survive == "Yes":
                    st.image("images/survived_yes.png", width=120)
                    st.session_state['user_prediction'] = "Survived"
                    survive_value = 1
                elif survive == "No":
                    st.image("images/survived_no.png", width=120)
                    st.session_state['user_prediction'] = "Did not survive"
                    survive_value = 0
                else:
                    st.image("images/survived_unknown.png", width=120)
                    survive_value = None

            if survive_value is not None:
                # Create a DataFrame for the selected person with the same structure as `personas.csv`
                selected_person_full = personas[personas['id'] == selected_id].copy()
                selected_person_full['Survived'] = survive_value

                # Save the selected person to a new CSV file with the same columns as `personas.csv`
                selected_person_save_path = 'data/selected_person.csv'
                if not os.path.exists('data'):
                    os.makedirs('data')
                selected_person_full.to_csv(selected_person_save_path, index=False)
                # st.success(f"Selected person's data with survival information has been saved to {selected_person_save_path}")
                # st.dataframe(selected_person_full)

                # Create a DataFrame for the selected person with survival information
                selected_person_user_friendly = user_friendly_personas[user_friendly_personas['id'] == selected_id].copy()

                # Map the numerical survival value to text
                survive_text = "Yes" if survive_value == 1 else "No"
                selected_person_user_friendly['Survived'] = survive_text

                # Remove the redundant 'survived' column
                if 'survived' in selected_person_user_friendly.columns:
                    selected_person_user_friendly = selected_person_user_friendly.drop(columns=['survived'])

                # Save only the selected person to a new CSV file
                user_friendly_personas_save_path = 'data/user_friendly_personas_with_survival.csv'
                if not os.path.exists('data'):
                    os.makedirs('data')
                selected_person_user_friendly.to_csv(user_friendly_personas_save_path, index=False)
                # st.success(f"User-friendly persona data with survival information has been saved to {user_friendly_personas_save_path}")
                # st.dataframe(selected_person_user_friendly)

            # Conditional rendering for feature importance sliders
            if survive in ["Yes", "No"]:
                st.write("How much do you think each factor has an influence on the survival outcome? Move the sliders to adjust the percentages.")
                survive = st.session_state['survive']
                st.write(f"Your prediction: {st.session_state['user_prediction']}")
                # Define a default value for the sliders
                default_value = 50 / 7.0  # Default value scaled to the slider range

                # Create sliders with default values
                st.session_state['sex_values'] = st.slider("Sex:", min_value=0, max_value=100, value=int(default_value * 7.0)) / 7.0
                st.session_state['embarked_values'] = st.slider("Embarked:", min_value=0, max_value=100, value=int(default_value * 7.0)) / 7.0
                st.session_state['passenger_class_values'] = st.slider("Passenger Class:", min_value=0, max_value=100, value=int(default_value * 7.0)) / 7.0
                st.session_state['age_values'] = st.slider("Age:", min_value=0, max_value=100, value=int(default_value * 7.0)) / 7.0
                st.session_state['sibsp_values'] = st.slider("Siblings / Spouse:", min_value=0, max_value=100, value=int(default_value * 7.0)) / 7.0
                st.session_state['parch_values'] = st.slider("Parents / Children:", min_value=0, max_value=100, value=int(default_value * 7.0)) / 7.0
                st.session_state['fare_values'] = st.slider("Fare:", min_value=0, max_value=100, value=int(default_value * 7.0)) / 7.0

                total_value = st.session_state['sex_values'] + st.session_state['embarked_values'] + st.session_state['passenger_class_values'] + st.session_state['age_values'] + st.session_state['sibsp_values'] + st.session_state['parch_values'] + st.session_state['fare_values']
                total_value = 100  # Ensure the total value is set to 100

                if st.button("Save answers"):
                    csv_file_path = 'data/pre_values.csv'

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


    elif pages.current == 1:
            st.header("AI Prediction and What-If Scenarios")
            survive = st.session_state['survive']
            survive_prediction = ':green[SURVIVED]' if st.session_state['user_prediction'] == "Survived" else ':red[DID NOT SURVIVE]'

            # Prepare data for prediction
            selected_person_full = pd.read_csv('data/selected_person.csv')
            selected_person_user_friendly = pd.read_csv('data/user_friendly_personas_with_survival.csv')
            selected_person_user_friendly = selected_person_user_friendly.drop(columns=['id'], errors='ignore')
            X_test_person = selected_person_full.drop(columns=['id', 'Survived'], errors='ignore')

            # Load and preprocess data
            data_loader = DataLoader()
            data_loader.load_dataset()
            data_loader.preprocess_data()

            X_train, y_train = data_loader.get_data_split()

            # Train Random Forest model
            rf = RandomForestClassifier()
            rf.fit(X_train, y_train)

            # Predict
            y_pred = rf.predict(X_test_person)
            y_pred_proba = rf.predict_proba(X_test_person)
            ai_prediction = ':green[SURVIVED]' if y_pred[0] == 1 else ':red[DID NOT SURVIVE]'

            # Convert probabilities to percentages
            survived_probability = y_pred_proba[0][1]
            did_not_survive_probability = y_pred_proba[0][0]
            survived_percentage = survived_probability * 100
            did_not_survive_percentage = did_not_survive_probability * 100

            # Define rectangle sizes
            survived_width = survived_percentage
            did_not_survive_width = did_not_survive_percentage

            ###### DISPLAYING THE PREDICTION RESULTS ######
            st.write(f"Here is the prediction for the selected person:")
            st.write(f"{st.session_state['selected_label']}")
            st.write("Your Data, featuring your prediction:")
            YourData.save_user_friendly_csv()

            # st.write("")
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
            st.write("Survived: 🟩   Did Not Survive: 🟥")
            st.write(f"Model Prediction: {ai_prediction}")
            st.write(f"Your Prediction: {survive_prediction}")
            if survive_prediction == ai_prediction:
                st.write('Your prediction was :green[CORRECT]!')
            else:
                st.write('Your prediction was :red[INCORRECT]!')
            




####### DICE COUNTERFACTUALS #############
            # Prepare data for DiCE
            # Drop 'id' column before passing to DiCE
            selected_person_full = selected_person_full.drop(columns=['id'], errors='ignore')

            # Ensure 'Survived' column is present
            if 'Survived' not in personas.columns:
                personas['Survived'] = y_pred[0]  # Add dummy value for consistency

            data_dice = dice_ml.Data(
                dataframe=selected_person_full,
                continuous_features=["Age", "Fare", "Pclass", "SibSp", "Parch"],
                outcome_name='Survived'
            )

            rf_dice = dice_ml.Model(model=rf, backend="sklearn")
            explainer = dice_ml.Dice(data_dice, rf_dice, method="random")

            # Extract input data point
            input_datapoint = selected_person_full.drop(columns=['Survived'])

            features_to_vary = ['Age', 'Fare', 'Pclass', 'SibSp', 'Parch']
            permitted_range = {
                'Age': [0, 80],
                'Fare': [1, 10000],
                'Pclass': [1, 3],
                'SibSp': [0, 6],
                'Parch': [0, 6]
            }
            
            # Generate counterfactuals
            cf = explainer.generate_counterfactuals(
                input_datapoint,
                total_CFs=3,
                desired_class="opposite",
                permitted_range=permitted_range,
                features_to_vary=features_to_vary,
                proximity_weight=1.5,
                diversity_weight=1.0,
                verbose=False
            )

            # Show counterfactual explanations
            if cf.cf_examples_list:
                cf_df = cf.cf_examples_list[0].final_cfs_df
                st.session_state.cf_df = cf_df

                def generate_textual_explanation(original_instance, counterfactual_instance, prediction_column='Survived'):
                    changes = []
                    for column in original_instance.index:
                        if column == prediction_column:
                            continue

                        original_value = original_instance[column]
                        new_value = counterfactual_instance[column]

                        # Convert to integers for categorical data
                        if column in ['Pclass', 'SibSp', 'Parch']:
                            original_value = int(original_value)
                            new_value = int(new_value)

                        # Round continuous features to the nearest integer and convert to int
                        elif column in ['Age', 'Fare']:
                            original_value = int(round(original_value))
                            new_value = int(round(new_value))

                        # Formatting for specific columns
                        if column == 'Fare':
                            original_value = f"£{original_value}"
                            new_value = f"£{new_value}"
                        elif column == 'SibSp':
                            original_value = f"{original_value} Sibling/Spouse" if original_value in [0, 1] else f"{original_value} Siblings"
                            new_value = f"{new_value} Sibling/Spouse" if new_value in [0, 1] else f"{new_value} Siblings"
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
                            original_value = f"{original_value}rd Passenger Class" if original_value == 3 else f"{original_value}st Passenger Class" if original_value == 1 else f"{original_value}nd Passenger Class"
                            new_value = f"{new_value}rd Passenger Class" if new_value == 3 else f"{new_value}st Passenger Class" if new_value == 1 else f"{new_value}nd Passenger Class"

                        if original_value != new_value:
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


                original_instance = selected_person_full.iloc[0]

                textual_explanations = []
                new_prediction_label = None
                for i in range(st.session_state.cf_df.shape[0]):
                    counterfactual_instance = st.session_state.cf_df.iloc[i]
                    changes_text, new_prediction_label = generate_textual_explanation(original_instance, counterfactual_instance)
                    textual_explanations.append(f"Changing {changes_text} will result in the prediction changing to '{new_prediction_label}'.")
                st.session_state[textual_explanations] = textual_explanations
                st.session_state[new_prediction_label] = new_prediction_label

        ################# COUNTERFACTUAL CONTAINER #################

                if new_prediction_label:

                    # st.write("Your Data:")
                    # YourData.save_user_friendly_csv()
                    st.write("\n")
                    
                # Create three columns for the scenarios
                cols = st.columns(3)
                for i in range(3):
                    with cols[i]:
                        # Create a container within each column
                        with st.container(height=300):
                            st.subheader(f"Scenario {i+1}:")
                            st.markdown(f"{textual_explanations[i]}")
    elif pages.current == 2:
        ######## EDIT DATA PAGE ########
        person = 'data/selected_person.csv'
        user_friendly_person = 'data/user_friendly_personas_with_survival.csv'

        st.write("Change data values:")

        # Check if the CSV file exists
        if os.path.exists(person):
            df1 = pd.read_csv(person)
        else:
            st.error("CSV file not found!")

        # Create a form for user input
        with st.form(key='edit_form'):
            st.write("Original Data")
            YourData.save_user_friendly_csv()

            sex = st.radio('Sex', options=['male', 'female'], key='Sex',horizontal=True)
            embarked = st.radio('Embarked', options=['France', 'England', 'Ireland'], key='Embarked',horizontal=True)
            pclass = st.number_input('Passenger Class', min_value=1, max_value=3, value=int(df1['Pclass'][0]), key='Pclass')
            age = st.number_input('Age', min_value=0, max_value=80, value=int(df1['Age'][0]), key='Age')
            sibsp = st.number_input('Siblings or Spouse', min_value=0, max_value=6, value=int(df1['SibSp'][0]), key='SibSp')
            parch = st.number_input('Parents or Children', min_value=0, max_value=6, value=int(df1['Parch'][0]), key='Parch')
            fare = st.number_input('Fare', min_value=0.00, max_value=10000.00, value=float(df1['Fare'][0]), key='Fare')
            survived = st.radio('Survived', options=['Yes', 'No'], key='Survived',horizontal=True)

            sex_male = None
            sex_female = None

            Embarked_C = None
            Embarked_Q = None
            Embarked_S = None

            if embarked == 'France':
                Embarked_C = 1
                Embarked_Q = 0
                Embarked_S = 0
            elif embarked == 'Ireland':
                Embarked_C = 0
                Embarked_Q = 1
                Embarked_S = 0
            else:
                Embarked_C = 0
                Embarked_Q = 0
                Embarked_S = 1

            if sex == 'male':
                sex_male = 1
                sex_female = 0
            else:
                sex_male = 0
                sex_female = 1


            if survived == 'Yes':
                survived = 1
            else:
                survived = 0

            # Submit button inside the form
            submit_button = st.form_submit_button(label='Re-run prediction')

            if submit_button:
                st.session_state.counter += 1
                st.write(f'The prediction button has been pressed {st.session_state.counter} times.')
                # Update the DataFrame with new values
                df1.loc[0, 'Pclass'] = pclass
                df1.loc[0, 'Age'] = age
                df1.loc[0, 'SibSp'] = sibsp
                df1.loc[0, 'Parch'] = parch
                df1.loc[0, 'Fare'] = fare
                df1.loc[0, 'Survived'] = survived
                df1.loc[0, 'Sex_male'] = sex_male
                df1.loc[0,'Sex_female'] = sex_female
                df1.loc[0, 'Embarked_C'] = Embarked_C
                df1.loc[0, 'Embarked_Q'] = Embarked_Q
                df1.loc[0, 'Embarked_S'] = Embarked_S


                # Save the updated DataFrame back to the CSV file
                df1.to_csv(person, index=False)

                st.success('Data updated successfully!')
                
                # Reload the updated data
                updated_df = pd.read_csv(person)
                st.write("Updated Data")
                YourData.save_user_friendly_csv()
                survive = st.session_state['survive']
                survive_prediction = ':green[SURVIVED]' if st.session_state['user_prediction'] == "Yes" else ':red[DID NOT SURVIVE]'
                # Prepare data for prediction
                selected_person_full = pd.read_csv('data/selected_person.csv')
                selected_person_user_friendly = pd.read_csv('data/user_friendly_personas_with_survival.csv')
                selected_person_user_friendly = selected_person_user_friendly.drop(columns=['id'], errors='ignore')
                X_test_person = selected_person_full.drop(columns=['id', 'Survived'], errors='ignore')

#################### RUN PREDICTION ON UPDATED DATA ####################

                # Load and preprocess data
                data_loader = DataLoader()
                data_loader.load_dataset()
                data_loader.preprocess_data()

                X_train, y_train = data_loader.get_data_split()

                # Train Random Forest model
                rf = RandomForestClassifier()
                rf.fit(X_train, y_train)

                # Predict
                y_pred = rf.predict(X_test_person)
                y_pred_proba = rf.predict_proba(X_test_person)
                ai_prediction = ':green[SURVIVED]' if y_pred[0] == 1 else ':red[DID NOT SURVIVE]'

                # Convert probabilities to percentages
                survived_probability = y_pred_proba[0][1]
                did_not_survive_probability = y_pred_proba[0][0]
                survived_percentage = survived_probability * 100
                did_not_survive_percentage = did_not_survive_probability * 100

                # Define rectangle sizes
                survived_width = survived_percentage
                did_not_survive_width = did_not_survive_percentage

                ###### DISPLAYING THE PREDICTION RESULTS ######
                st.write(f"Here is the prediction for the updated data:")
                

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
                st.write("Survived: 🟩   Did Not Survive: 🟥")
                st.write(f"Model Prediction: {ai_prediction}")
                st.write(f"Your Prediction: {survive_prediction}")
                if survive_prediction == ai_prediction:
                    st.write('Your prediction was :green[CORRECT]!')
                else:
                    st.write('Your prediction was :red[INCORRECT]!')
                

        st.write(":orange[Once you're done with exploring the data, proceed to the final page.]")


    elif pages.current == 3:
            ########## FINAL PAGE ##########
            st.write("Page 4")
            with st.form(key='edit_form'):
                st.write("How much do you think each factor has an influence on the survival outcome?")
                    
                # Retrieve and store each value in session state
                default_value = 50 / 7.0

                st.session_state['sex_values_post'] = st.slider("Sex:", min_value=0, max_value=100, value=int(default_value * 7.0)) / 7.0
                st.session_state['embarked_values_post'] = st.slider("Embarked:", min_value=0, max_value=100, value=int(default_value * 7.0)) / 7.0
                st.session_state['passenger_class_values_post'] = st.slider("Passenger Class:", min_value=0, max_value=100, value=int(default_value * 7.0)) / 7.0
                st.session_state['age_values_post'] = st.slider("Age:", min_value=0, max_value=100, value=int(default_value * 7.0)) / 7.0
                st.session_state['sibsp_values_post'] = st.slider("Siblings / Spouse:", min_value=0, max_value=100, value=int(default_value * 7.0)) / 7.0
                st.session_state['parch_values_post'] = st.slider("Parents / Children:", min_value=0, max_value=100, value=int(default_value * 7.0)) / 7.0
                st.session_state['fare_values_post'] = st.slider("Fare:", min_value=0, max_value=100, value=int(default_value * 7.0)) / 7.0

                submit_button = st.form_submit_button(label='Save your answers')
                if submit_button:
                    # Define CSV file path
                    csv_file_path = 'data/post_values.csv'
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
            
            import plotly.express as px
            # Load and preprocess data
            data_loader = DataLoader()
            data_loader.load_dataset()
            data_loader.preprocess_data()

            X_train, y_train = data_loader.get_data_split()

            # Train Random Forest model
            
            rf.fit(X_train, y_train)

        # Calculate feature importances
            importances = rf.feature_importances_
            feature_names = X_train.columns

            # Create a DataFrame for feature importances
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            })
            feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

            # Define label maps
            label_mapping = {
                'Sex_female': 'Female',
                'Sex_male': 'Male',
                'Embarked_C': 'France',
                'Embarked_Q': 'Ireland',
                'Embarked_S': 'England',
                'Pclass': 'Passenger Class',
                'Age': 'Age',
                'SibSp': 'Siblings / Spouse',
                'Parch': 'Parents / Children',
                'Fare': 'Fare'
            }
            label_mapping_input = {'fare_values': 'Fare', 
                                   'age_values': 'Age', 
                                   'passenger_class_values': 'Passenger Class', 
                                   'sibsp_values': 'Siblings / Spouse', 
                                   'parch_values': 'Parents / Children',
                                   'sex_values': 'Sex',
                                   'embarked_values': 'Embarked'
                                   }
            
            label_mapping_input_post = {'fare_values_post': 'Fare',
                                        'age_values_post': 'Age',
                                        'passenger_class_values_post': 'Passenger Class',
                                        'sibsp_values_post': 'Siblings / Spouse',
                                        'parch_values_post': 'Parents / Children',
                                        'sex_values_post': 'Sex',
                                        'embarked_values_post': 'Embarked'
                                        }


        # Map feature names to user-friendly labels
            feature_importance_df['feature'] = feature_importance_df['feature'].map(label_mapping).fillna(feature_importance_df['feature'])
            feature_importance_df.to_csv('feature_importance.csv', index=False)
            # Define color mapping for the user-friendly labels
            colour_mapping = {
                'Fare': '#c492f5',    # Purple
                'Age': '#fff32c',     # Yellow
                'Passenger Class': '#ff2c2c',  # Red
                'Siblings / Spouse': '#69e461',   # Green
                'Parents / Children': '#6b8afa',    # Blue
                'Female': '#ff8f3d', # Orange
                'Male': '#ff8f3d', # Orange
                'France': '#ff63a9', # Pink
                'England': '#ff63a9', # Pink
                'Ireland': '#ff63a9', # Pink
            }
            
            colour_mapping_input = {
                'Fare': '#c492f5',    # Purple
                'Age': '#fff32c',     # Yellow
                'Passenger Class': '#ff2c2c',  # Red
                'Siblings / Spouse': '#69e461',   # Green
                'Parents / Children': '#6b8afa',    # Blue
                'Sex': '#ff8f3d', # Orange
                'Embarked': '#ff63a9', # Pink
            }
            
            colour_mapping_input_post = {
                'Fare': '#c492f5',    # Purple
                'Age': '#fff32c',     # Yellow
                'Passenger Class': '#ff2c2c',  # Red
                'Siblings / Spouse': '#69e461',   # Green
                'Parents / Children': '#6b8afa',    # Blue
                'Sex': '#ff8f3d', # Orange
                'Embarked': '#ff63a9', # Pink
            }

            # Map user-friendly labels to colors
            feature_importance_df['color'] = feature_importance_df['feature'].map(colour_mapping).fillna('#808080')
            
            input_feature_names = ['sex_values', 'embarked_values', 'passenger_class_values', 'age_values', 'sibsp_values', 'parch_values', 'fare_values']
            input_feature_names_post = ['sex_values_post', 'embarked_values_post', 'passenger_class_values_post', 'age_values_post', 'sibsp_values_post', 'parch_values_post', 'fare_values_post']
            
            # Retrieve the stored slider values from st.session_state
            input_feature_values = [
                st.session_state['sex_values'],
                st.session_state['embarked_values'],
                st.session_state['passenger_class_values'],
                st.session_state['age_values'],
                st.session_state['sibsp_values'],
                st.session_state['parch_values'],
                st.session_state['fare_values']
            ]
            
            input_feature_values_post = [
                st.session_state['sex_values_post'],
                st.session_state['embarked_values_post'],
                st.session_state['passenger_class_values_post'],
                st.session_state['age_values_post'],
                st.session_state['sibsp_values_post'],
                st.session_state['parch_values_post'],
                st.session_state['fare_values_post']
            ]
            
            # Create DataFrames for user inputs
            feature_importance_input_df = pd.DataFrame({
                'feature': [label_mapping_input[name] for name in input_feature_names],
                'importance': input_feature_values
            })
            
            feature_importance_input_df_post = pd.DataFrame({
                'feature': [label_mapping_input_post[name] for name in input_feature_names_post],
                'importance': input_feature_values_post
            })

            # Create the Plotly pie charts
            fig = px.pie(feature_importance_df, 
                        values='importance', 
                        names='feature',
                        color='feature',
                        color_discrete_map=colour_mapping,
                        title='Feature Importance for the Dataset')
            
            fig_input = px.pie(feature_importance_input_df,
                            values='importance', 
                            names='feature',
                            color='feature',
                            color_discrete_map=colour_mapping_input,
                            title='Your Predicted Feature Importance')
            
            fig_input_post = px.pie(feature_importance_input_df_post,
                                    values='importance',
                                    names='feature',
                                    color='feature',
                                    color_discrete_map=colour_mapping_input_post,
                                    title='Your Predicted Feature Importance (Post)')

            # Display the plots in Streamlit
            st.write("See feature importance")
            
            with st.expander("Features pre experiment"):
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig)
                with col2:
                    st.plotly_chart(fig_input)
            
            with st.expander("Features post experiment"):
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig)
                with col2:
                    st.plotly_chart(fig_input_post)
    elif pages.current == 4:
        

            # Set the title of the Streamlit app
            st.title("Digital UI User Experience Questionnaire")

            # Embed the Google Form using the iframe URL
            st.components.v1.iframe(
                src="https://docs.google.com/forms/d/e/1FAIpQLSe2yIAbM30hncvXZwhPbFjf21paMnSxNnE0Vw5IYsfh1Q7s1A/viewform?embedded=true", 
                width=640, 
                height=1260, 
                scrolling=True
            )


