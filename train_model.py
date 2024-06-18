import spacy
from spacy.training import Example
from spacy.training import offsets_to_biluo_tags
import random
import pandas as pd


def read_input_data():
    # Replace 'your_file.csv' with the path to your CSV file
    df = pd.read_csv('medical_training_data.csv', sep='|')
    return df

# Step 2: Prepare Training Data
def prepare_training_data(df):
    formatted_data = []
    for index, row in df.iterrows():
        # Access data in each row using column names
        text = row['Text']
        entities = [(row['BeginOffset'], row['EndOffset'], row['Category'])]
        formatted_data.append((text, {"entities": entities}))
    return formatted_data


df = read_input_data()
formatted_training_data = prepare_training_data(df)

# Step 3: Define Training Configuration
labels = ["BLOOD_PRESSURE_READING", "DATE", "DATE_OF_BIRTH"]
iterations = 20

# Step 4: Initialize NER Model and Trainer
nlp = spacy.blank("en")
ner = nlp.add_pipe("ner", last=True)
for label in labels:
    ner.add_label(label)

# Initialize the optimizer
optimizer = nlp.begin_training()

# Step 5: Train the Model
for itn in range(iterations):
    random.shuffle(formatted_training_data)
    losses = {}
    for text, annotations in formatted_training_data:
        doc = nlp.make_doc(text)
        # Check entity alignment and convert to BILUO tags (optional)
        biluo_tags = offsets_to_biluo_tags(doc, annotations['entities'])
        # print(text)
        # print(biluo_tags)
        example = Example.from_dict(doc, annotations)
        nlp.update([example], drop=0.5, losses=losses)
    print(f"Iteration {itn + 1}: Losses: {losses}")

# Step 6: Save the Model
output_dir = "output_model"
nlp.to_disk(output_dir)
print(f"Model saved to {output_dir}")



