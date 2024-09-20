import streamlit as st
import pandas as pd
import os
class YourData:
    @staticmethod
    def save_user_friendly_csv():
        one_hot_encoded_file = '/Users/miry/Documents/repos/CF-experimentation/data/arduino_data.csv'
        user_friendly_file = '/Users/miry/Documents/repos/CF-experimentation/data/user_friendly_data.csv'

        if os.path.exists(one_hot_encoded_file):
            df = pd.read_csv(one_hot_encoded_file)  # Load your one-hot encoded data
            # Define reverse mapping for columns
            reverse_mapping = {
                'Sex_female': 'Sex',
                'Sex_male': 'Sex',
                'Embarked_C': 'Embarked',
                'Embarked_Q': 'Embarked',
                'Embarked_S': 'Embarked',
                'Pclass': 'Passenger Class',
                'Age': 'Age',
                'SibSp': 'Siblings / Spouse',
                'Parch': 'Parents / Children',
                'Fare': 'Fare',
                'Survived': 'Survived'
            }
            # Define the mappings
            embarked_mapping = {
                'Embarked_C': 'France',
                'Embarked_Q': 'Ireland',
                'Embarked_S': 'England'
            }
            
            survived_mapping = {
                1: 'Yes',
                0: 'No'
            }
            
            # Convert one-hot encoded columns to their original categories
            df['Sex'] = df[['Sex_female', 'Sex_male']].idxmax(axis=1).str.replace('Sex_', '')
            df['Embarked'] = df[['Embarked_C', 'Embarked_Q', 'Embarked_S']].idxmax(axis=1).map(embarked_mapping)
            
            # Convert Survived from 1/0 to Yes/No
            df['Survived'] = df['Survived'].map(survived_mapping)
            df['Fare'] = df['Fare'].apply(lambda x: f'£{x:.2f}')
            
            # Drop the one-hot encoded columns
            df = df[['Sex', 'Embarked', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']]
            
            # Rename columns to match the user-friendly format
            df.columns = ['Sex', 'Embarked', 'Passenger Class', 'Age', 'Siblings / Spouse', 'Parents / Children', 'Fare', 'Survived']
            
            # Save to CSV
            df.to_csv(user_friendly_file, index=False)

            # Check if the file exists after saving
            if os.path.exists(user_friendly_file):
                # Load the data from the user-friendly CSV file
                df = pd.read_csv(user_friendly_file)
                
                # Define the color map
                colour_map = {
                    'Fare': '#c492f5',    # Purple
                    'Age': '#fff32c',     # Yellow
                    'Passenger Class': '#ff2c2c',  # Red
                    'Siblings / Spouse': '#69e461',   # Green
                    'Parents / Children': '#6b8afa',    # Blue
                    'Sex': '#ff8f3d', # Orange
                    'Embarked': '#ff63a9', # Pink
                }

                # Define a function to color cells based on the column and make text bold
                def color_cells(val, column_name):
                    color = colour_map.get(column_name, '#ffffff')  # Default to white if column not in colour_map
                    return f'background-color: {color};color: black; font-weight: bold'

                # Define a function to style the "Survived" column
                def color_survived(val):
                    if val == 'Yes':
                        return 'background-color: white; color: black; font-weight: bold'
                    else:
                        return 'background-color: black; color: white; font-weight: bold'

                # Apply color styling to each column based on the colour_map and make text bold
                styled_df = df.style.applymap(lambda val: color_cells(val, 'Fare'), subset=['Fare']) \
                                    .applymap(lambda val: color_cells(val, 'Age'), subset=['Age']) \
                                    .applymap(lambda val: color_cells(val, 'Passenger Class'), subset=['Passenger Class']) \
                                    .applymap(lambda val: color_cells(val, 'Siblings / Spouse'), subset=['Siblings / Spouse']) \
                                    .applymap(lambda val: color_cells(val, 'Parents / Children'), subset=['Parents / Children']) \
                                    .applymap(lambda val: color_cells(val, 'Sex'), subset=['Sex']) \
                                    .applymap(lambda val: color_cells(val, 'Embarked'), subset=['Embarked']) \
                                    .applymap(color_survived, subset=['Survived']) \
                                    .set_table_styles([{'selector': 'td', 'props': [('font-weight', 'bold')]}])  # Set all cells to bold

                # Display the styled DataFrame in Streamlit
                st.dataframe(styled_df, hide_index=True)
            else:
                st.error("User-friendly CSV file not found.")
        else:
            st.error("One-hot encoded data file not found.")

    def closest_passenger():
        one_hot_encoded_file = '/Users/miry/Documents/repos/CF-experimentation/data/titanic_edited.csv'
        user_friendly_file = '/Users/miry/Documents/repos/CF-experimentation/data/titanic_user_friendly_data.csv'

        if os.path.exists(one_hot_encoded_file):
            df = pd.read_csv(one_hot_encoded_file)  # Load your one-hot encoded data
            # Define reverse mapping for columns
            reverse_mapping = {
                'Pclass': 'Passenger Class',
                'SibSp': 'Siblings / Spouse',
                'Parch': 'Parents / Children',
            }
            # Define the mappings
            embarked_mapping = {
                'C': 'France',
                'Q': 'Ireland',
                'S': 'England'
            }
            
            survived_mapping = {
                1: 'Yes',
                0: 'No'
            }
        
            
            # Convert Survived from 1/0 to Yes/No
            df['Embarked'] = df['Embarked'].map(embarked_mapping)
            df['Survived'] = df['Survived'].map(survived_mapping)
            df['Fare'] = df['Fare'].apply(lambda x: f'£{x:.2f}')
            
            # Drop the one-hot encoded columns
            df = df[['Name','Sex', 'Embarked', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']]
            
            # Rename columns to match the user-friendly format
            df.columns = ['Name','Sex', 'Embarked', 'Passenger Class', 'Age', 'Siblings / Spouse', 'Parents / Children', 'Fare', 'Survived']
            
            # Save to CSV
            df.to_csv(user_friendly_file, index=False)

            # Check if the file exists after saving
            if os.path.exists(user_friendly_file):
                # Load the data from the user-friendly CSV file
                df = pd.read_csv(user_friendly_file)
                
                # Define the color map
                colour_map = {
                    'Fare': '#c492f5',    # Purple
                    'Age': '#fff32c',     # Yellow
                    'Passenger Class': '#ff2c2c',  # Red
                    'Siblings / Spouse': '#69e461',   # Green
                    'Parents / Children': '#6b8afa',    # Blue
                    'Sex': '#ff8f3d', # Orange
                    'Embarked': '#ff63a9', # Pink
                }

                # Define a function to color cells based on the column and make text bold
                def color_cells(val, column_name):
                    color = colour_map.get(column_name, '#ffffff')  # Default to white if column not in colour_map
                    return f'background-color: {color};color: black; font-weight: bold'

                # Define a function to style the "Survived" column
                def color_survived(val):
                    if val == 'Yes':
                        return 'background-color: white; color: black; font-weight: bold'
                    else:
                        return 'background-color: black; color: white; font-weight: bold'

                # Apply color styling to each column based on the colour_map and make text bold
                styled_df = df.style.applymap(lambda val: color_cells(val, 'Fare'), subset=['Fare']) \
                                    .applymap(lambda val: color_cells(val, 'Age'), subset=['Age']) \
                                    .applymap(lambda val: color_cells(val, 'Passenger Class'), subset=['Passenger Class']) \
                                    .applymap(lambda val: color_cells(val, 'Siblings / Spouse'), subset=['Siblings / Spouse']) \
                                    .applymap(lambda val: color_cells(val, 'Parents / Children'), subset=['Parents / Children']) \
                                    .applymap(lambda val: color_cells(val, 'Sex'), subset=['Sex']) \
                                    .applymap(lambda val: color_cells(val, 'Embarked'), subset=['Embarked']) \
                                    .applymap(color_survived, subset=['Survived']) \
                                    .set_table_styles([{'selector': 'td', 'props': [('font-weight', 'bold')]}])  # Set all cells to bold

                # Display the styled DataFrame in Streamlit
                st.dataframe(styled_df, hide_index=True)
            else:
                st.error("User-friendly CSV file not found.")
        else:
            st.error("One-hot encoded data file not found.")
# YourData.closest_passenger()
