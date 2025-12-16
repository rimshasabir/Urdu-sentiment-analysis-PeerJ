import pandas as pd
from LughaatNLP import LughaatNLP
import stanza

# Instance Calling
urdu_text_processing = LughaatNLP()

# Read the CSV file
data = pd.read_csv('C:\\Users\\RIMSHA\\PycharmProjects\\RESEARCH\\5.csv')

# Replace numerical digits with Urdu digits
data['OriginalTweet'] = data['OriginalTweet'].astype(str).apply(urdu_text_processing.replace_digits)

# Remove special characters except question marks
data['OriginalTweet'] = data['OriginalTweet'].apply(urdu_text_processing.remove_special_characters_exceptUrdu)

# Remove white and extra spaces
data['SpaceText'] = data['OriginalTweet'].apply(urdu_text_processing.punctuations_space)

# Remove white and white spaces
data['SpaceText'] = data['SpaceText'].apply(urdu_text_processing.remove_whitespace)

# Tokenize the cleaned text
data['TokenizedText'] = data['SpaceText'].apply(urdu_text_processing.urdu_tokenize)

# Initialize Stanza pipeline for Urdu
stanza.download('ur')  # Ensure Urdu model is downloaded
nlp = stanza.Pipeline('ur')

# Function to perform POS tagging
def pos_tagging(text):
    doc = nlp(text)
    return [(word.text, word.xpos) for sentence in doc.sentences for word in sentence.words]

# Apply POS tagging
data['POSTags'] = data['SpaceText'].apply(pos_tagging)

# Print the tokenized text and POS tags in a more organized manner
# Save the DataFrame to a CSV file with UTF-8 encoding
output_csv_file = 'C:\\Users\\RIMSHA\\PycharmProjects\\RESEARCH\\output.csv'
data.to_csv(output_csv_file, index=False, encoding='utf-8')

print(f"Data saved to {output_csv_file} with Urdu text preserved.")