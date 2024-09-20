import streamlit as st
import csv

st.title("About this study")

st.write("You are being invited to take part in a research study. Before you decide whether or not to take part, it is important for you to understand why the research is being done and what it will involve. Please take time to read the following information carefully.")

st.write("**What is the purpose of the study?**")
st.write("The aim of this study is to explore how participants engage with machine learning algorithms, specifically using the Titanic Dataset, through interactive activities. The goal is to gain insights into how interactive learning approaches can enhance understanding and engagement with machine learning concepts.")

st.write("**Do I have to take part?**")
st.write("It is up to you to decide whether or not to take part in this research study. If you do decide to take part you will be given this information sheet along with a privacy notice that will explain how your data will be collected and used, and be asked to give your consent. If you decide to take part you are still free to withdraw at any time and without giving a reason.")

st.write("**What will happen to me if I take part?**")
st.write("You will be asked to provide a brief overview of your existing understanding of machine learning before the start of the experiment.  Your responses will be recorded for qualitative analysis. Using the Titanic Dataset, you will use a Random Forest Classifier algorithm to predict the survival outcome of a hypothetical persona onboard the Titanic. This involves scanning a selection of data cards on an RFID reader to construct a dataset representing various attributes of the persona. Following the prediction made by the algorithm, you will have the opportunity to discuss the alignment of your initial prediction with that of the AI's, as well as your thoughts on the outcome. These discussions will be recorded for qualitative analysis purposes. You will receive three alternative scenarios, generated by DiCE Counterfactual explanations, where the opposite prediction is true. Each scenario will offer insights into the changes required in the dataset to achieve a different outcome. Subsequently, you will be encouraged to modify your dataset based on the provided counterfactual explanations and interact with the model again to observe any alterations in the prediction outcome. Finally, you will be presented with a series of questions regarding your comprehension of the model's prediction process and any general observations you may have. Your responses will be recorded for qualitative analysis. The study should take around 30 minutes from start to finish.")

st.write("**What are the possible disadvantages and risks of taking part?**") 
st.write("There are no identified risks in partaking in this study.")

st.write("**What are the possible benefits of taking part?**")
st.write("Participating in this study offers a unique opportunity to learn, contribute to research, and develop valuable skills in the domain of machine learning and predictive modelling. Engaging with machine learning algorithms and datasets can deepen your understanding of how these technologies work and how they are applied in real-world scenarios. You will gain practical experience in constructing datasets, making predictions, and analysing results, which can be valuable for individuals interested in data science or machine learning.")

st.write("**Will what I say in this study be kept confidential?**")
st.write("All information collected will be kept strictly confidential and personal data anonymised. Data will be kept on the UAL OneDrive and UAL GitHub and deleted after the research student graduates. You will be provided with a Participant ID, which you can quote if you wish to withdraw your data.")

st.write("**What should I do if I want to take part?**")
st.write("To opt in in this study, you should inform the student conducting the research and complete the consent form provided.")

st.write("**What will happen to the results of the research study?**")
st.write("The results of this study will be used in the student's thesis for their MRes Creative Computing degree. A copy of the findings will be offered to each participant upon completion of the study.")

st.write("**Your Right to Withdraw from the study**")
st.write("You can withdraw from the study at any point, you are not required to give a reason to do so and all of your data will be deleted from the study. Quote the participant ID given to you at the start of the study when you contact us.")

st.write("**Who is organising the research?**")
st.write("This research is conducted by a student of MRes Creative Computing at the Creative Computing Institute in University of the Arts London under the supervision of Prof. Nick Bryan-Kinns")

st.write("**Who has reviewed the study?**")
st.write("The research has been approved by Dr Bea Wohl in line with the UAL ethics policy")


st.title("Consent Form")

with st.form("consent_form"):
    st.write("Please read the following statements carefully and confirm your consent by checking the boxes below.")
    
    checkbox_val = st.checkbox("**I understand that** I have given my consent for being interviewed by the researcher and for answering a questionnaire on my understanding of machine learning and AI.", key="checkbox1")
    checkbox_val1 = st.checkbox("**I understand that** I have given my consent for being interviewed by the researcher and for answering a questionnaire on my understanding of machine learning and AI.", key="checkbox2")
    checkbox_val2 = st.checkbox("**I understand that** I have given my consent for my voice to be recorded during the interview and interactive part of the experiment.", key="checkbox3")
    checkbox_val3 = st.checkbox("**I understand that** I have given my consent for my answers to be recorded during the interactive part of the experiment.", key="checkbox4")
    checkbox_val4 = st.checkbox("**I understand that** I have given approval for my opinion to be included / published in the final outcome of this research project and to be used in future research.", key="checkbox5")
    checkbox_val5 = st.checkbox("**I understand that** my involvement in this study, and particular data from this research (name and contact telephone number), will remain strictly confidential. My personal details will be anonymised and only the researchers involved in the study will have access to the data. It has been explained to me that the data will be kept in file, in case of possible use for future research, once the experimental program has been completed.", key="checkbox6")
    checkbox_val6 = st.checkbox("**I understand that** the identifiable data will be shared with the researcher's Academic Supervisors and may be monitored by the researcher's College Research Committee (CRC) and the 'University of the Arts London' Ethics Sub-Committee.", key="checkbox7")
    checkbox_val7 = st.checkbox("**I have read the information sheet** about the research project in which I have been asked to take part and have been given a copy of this information sheet to keep.", key="checkbox8")
    checkbox_val8 = st.checkbox("What is going to happen and why it is being done has been explained to me, and I have had the opportunity to discuss the details and ask questions.", key="checkbox9")
    checkbox_val9 = st.checkbox("Having given this consent I understand that I have the right to withdraw from the study at any time without disadvantage to myself and without having to give any reason.", key="checkbox10")
    checkbox_val0 = st.checkbox("**I hereby fully and freely consent to participation in the study which has been fully explained to me.**", key="checkbox11")
    
    signed = st.text_input("Participant's Name:", key="signed_name")

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
    # Data to be saved in CSV
        data = {
            "Participant": signed,
            "Consent1": checkbox_val,
            "Consent2": checkbox_val1,
            "Consent3": checkbox_val2,
            "Consent4": checkbox_val3,
            "Consent5": checkbox_val4,
            "Consent6": checkbox_val5,
            "Consent7": checkbox_val6,
            "Consent8": checkbox_val7,
            "Consent9": checkbox_val8,
            "Consent10": checkbox_val9,
            "Consent11": checkbox_val0
        }

        # Saving data to a CSV file
        with open('data/consent_data.csv', 'a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=data.keys())
            # If file is empty, write the header
            if file.tell() == 0:
                writer.writeheader()
            writer.writerow(data)

        st.write("Thank you for your participation. Your consent has been recorded.")
st.write("Student investigator’s name _Miryana Ivanova_")
st.write("Email: m.ivanova0720231@arts.ac.uk")
st.write("_Thank you_")