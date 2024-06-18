# NER_Medical_Data_Extraction

Training Custom AI model - Named Entity Recognition (NER)
Annotating text using NER model - NLPWhat is Named Entity Recognition (NER) ?
Named Entity Recognition (NER) is a subtask of information extraction that seeks to locate and classify named entities mentioned in unstructured text into predefined categories such as the names of persons, organizations, locations, expressions of times, quantities, monetary values, percentages, etc.
Purpose:
Information Extraction: To extract meaningful information from text.
Data Organization: To help organize and structure data.
Improved Search: To enhance search capabilities by recognizing and indexing named entities.

How It Works:
Tokenization: The text is split into individual words or tokens.
Part-of-Speech Tagging: Each token is tagged with its part of speech.
Entity Detection: Algorithms and models identify tokens that correspond to named entities.
Classification: Detected entities are classified into predefined categories.

Common Categories:
Person: Names of people (e.g., "John Doe").
Organization: Names of companies, institutions, etc. (e.g., "Google").
Location: Geographical locations (e.g., "New York").
Date/Time: Expressions of date and time (e.g., "January 1, 2024").
Monetary Values: Amounts of money (e.g., "$100").

Example:
Text: "Apple Inc. was founded by Steve Jobs and Steve Wozniak in Cupertino, California."
NER Output:
Organization: Apple Inc.
Person: Steve Jobs, Steve Wozniak
Location: Cupertino, California

Applications:
Search Engines: Enhance search results by recognizing named entities.
Content Categorization: Automatically tag and categorize content.
Customer Service: Improve response by understanding and extracting relevant information from queries.
Financial Analysis: Extract key financial information from reports.

Tools and Libraries:
spaCy: An open-source library for advanced NLP in Python.
NLTK: Natural Language Toolkit, a suite of libraries and programs for symbolic and statistical natural language processing.
Stanford NER: A Java-based named entity recognizer.
OpenNLP: Apache OpenNLP library for machine learning based processing of natural language text.

How to train your custom AI model to extract entity?
In this example, we'll extract BP Reading from the medical notes of the doctor using Spacy module in Python. To proceed, create a virtual environment in python and activate the virtual environment using the below commands.
```bash
python -m venv ner_venv
ner_venv\Scripts\activate
```

Once the virtual environment is activated, install the spacy module. This example is based on the Open-Source spacy module. 
pip install spacy
Once the package is installed, we need to create a sample dataset which we will consume to train the model. For this example, let me create some dummy text and add the span of the text that needs to be labelled. 
In natural language processing (NLP), "spanning" refers to the range of text covered by a particular linguistic unit, such as a named entity, phrase, or sentence. It typically involves identifying the start and end positions (offsets) of the text segment that corresponds to the unit of interest.
Let me create a sample csv file and name it as medical_training_data.csv. Populate some medical text samples that needs to annotated along with beginning offset of the entity, end offset of the entity, custom label that you want to use. In this case, I am going to annotate BP reading from the text and label it as "BLOOD_PRESSURE_READING". My sample dataset will look something like this. I used "|" symbol to separate the columns.
Text|BeginOffset|EndOffset|Category
Physical examination revealed elevated blood pressure (140/90 mmHg)|55|66|BLOOD_PRESSURE_READING
Patient visited with BP reading 120/80 mmHg. Symptoms include dizziness and fatigue.|32|43|BLOOD_PRESSURE_READING
BP recorded as 135 over 90 during the last check-up. Patient complains of occasional headaches.|15|26|BLOOD_PRESSURE_READING
Today's BP reading is 130/85. Patient reports feeling lightheaded and short of breath.|22|28|BLOOD_PRESSURE_READING
Patient recent BP reading shows 120/90 mmHg at 10 AM|32|38|BLOOD_PRESSURE_READING
Now, we need to read this data for pre-processing. We will use panda library to read the data for this purpose.
pip install pandas
pip install numpy==1.26.4
Make sure to install numpy 1.26.4 version. Else you might face incompatibility issues with error.
We'll create a function to read the csv file and return as data frame.
def read_input_data():
    # Replace 'your_file.csv' with the path to your CSV file
    df = pd.read_csv('medical_training_data.csv', sep='|')
    return df

df = read_input_data()
Next step is to prepare the training data. Spacy module requires data to be in the tuple format like this [(beginOffset, endOffset, label)]. So we prepare the data to this format using the below function.
def prepare_training_data(df):
    formatted_data = []
    for index, row in df.iterrows():
        # Access data in each row using column names
        text = row['Text']
        entities = [(row['BeginOffset'], row['EndOffset'], row['Category'])]
        formatted_data.append((text, {"entities": entities}))
    return formatted_data

formatted_training_data = prepare_training_data(df)
Once the data (in expected format) is ready for training, set the iteration count and the training configurations. 
labels = ["BLOOD_PRESSURE_READING"]
iterations = 20
During training, the model will learn to identify and classify text segments that correspond to these labels. Refers to how many times the model will work through the entire training dataset. In each iteration, the model adjusts its weights based on the training data to improve its predictions. More iterations can lead to better performance as the model has more opportunities to learn from the data. However, too many iterations can lead to overfitting, where the model performs well on the training data but poorly on new, unseen data.
nlp = spacy.blank("en") # Starting with a blank model allows for a fully customized NER training process.
ner = nlp.add_pipe("ner", last=True) #This adds the NER component to the pipeline of the blank spaCy model
for label in labels:
    ner.add_label(label)

optimizer = nlp.begin_training()
Parameter (last=True): Ensures that the NER component is added as the last component in the pipeline.
Optimizer: In machine learning, an optimizer adjusts the weights of the model based on the computed gradients during training to minimize the loss.
for itn in range(iterations):
    random.shuffle(formatted_training_data)
    losses = {}
    for text, annotations in formatted_training_data:
        doc = nlp.make_doc(text)
        # Check entity alignment and convert to BILUO tags (optional)
        biluo_tags = offsets_to_biluo_tags(doc, annotations['entities'])
        example = Example.from_dict(doc, annotations)
        nlp.update([example], drop=0.5, losses=losses)
    print(f"Iteration {itn + 1}: Losses: {losses}")
The outer loop runs for a specified number of iterations (iterations), shuffling the training data at the start of each iteration to ensure the model doesn't learn the order of the data, which helps in generalizing better.
Losses Dictionary: A dictionary to keep track of the losses for each iteration.
Inner for loop iterates through each training example in the shuffled formatted_training_data.
Creating a Doc Object: Converts the raw text into a spaCy Doc object, which is the standard way spaCy represents a text and its annotations.
BILUO Tags Conversion (Optional): Converts the entity annotations from character offsets to BILUO tags (Begin, Inside, Last, Unit, Outside). This step is optional but useful for debugging and ensuring the entity annotations align correctly with the tokens in the Doc object.
Creating an Example: Converts the Doc object and its annotations into a spaCy Example object. The Example object is the standard format spaCy uses for training.
Updating the Model: Updates the model using the current training example.
[example]: A list containing the current example to update the model with.
drop=0.5: Specifies the dropout rate, which is a regularization technique used to prevent overfitting by randomly dropping units during training. A dropout rate of 0.5 means that 50% of the units are dropped.
losses=losses: The losses dictionary to store the loss values for the current update step.

output_dir = "output_model"
nlp.to_disk(output_dir)
print(f"Model saved to {output_dir}")
Above code saves the model to the output directory. Once the code is done, execute the program to generate a output model that we trained.
Deploying the model to get NER output for evaluation data:-
Create a new python file named deploy.py and add the below code.
import spacy

model_path = 'output_model'  # Update with the path to your saved model
nlp_cm = spacy.load(model_path)

# Process test text
text = """
DOB: 17/12/1994
Subjective:
Patient presents today for a follow-up visit regarding chronic hypertension. Reports occasional headaches but denies any chest pain, shortness of breath, or dizziness. Compliant with prescribed medications.

Objective:

Vitals: BP readings consistently elevated around 120/90 mmHg during office visits.
General: Patient appears well-nourished and in no acute distress.
CV (Cardiovascular): Regular rate and rhythm, no murmurs.
Neurological: No focal deficits noted.
"""
doc_cm = nlp_cm(text)
for ent in doc_cm.ents:
    print("{} - {}".format(ent.text, ent.label_))

In the above code, I loaded the output_model that we created from the previous code. I provided sample medical notes text. From this note, my objective is to extract the Blood Pressure Reading. Once executed it will provide the following result.
120/90 mmHg - BLOOD_PRESSURE_READING
Similarly, we can annotate multiple labels to extract multiple entities from the document.
