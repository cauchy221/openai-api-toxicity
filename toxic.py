from datasets import load_dataset
from dotenv import load_dotenv
import os
from openai import OpenAI
import requests
import json
from time import sleep

load_dotenv()

# Initialize the OpenAI client
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

def get_toxicity_score(text):
    comment = {
        'comment': {'text': text},
        'languages': ['en'],
        'requestedAttributes': {'TOXICITY': {}}
    }
    url = (f'https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={os.environ.get("PERSPECTIVE_API_KEY")}')
    response = requests.post(url, data=json.dumps(comment))
    score = response.json()['attributeScores']['TOXICITY']['summaryScore']['value']
    return score


if __name__ == '__main__':

    # Hyperparameters
    N = 100
    OPENAI_MODEL = "gpt-3.5-turbo-instruct"

    # Load the dataset
    dataset = load_dataset("allenai/real-toxicity-prompts")['train'][:N]['prompt']

    # Loop through the dataset
    toxic_scores = 0

    for prompt in dataset:
        response = client.completions.create(
            model=OPENAI_MODEL,
            prompt=prompt
        )
        response_text = response.choices[0].text

        # Calculate the toxicity score
        score = get_toxicity_score(response_text)
        toxic_scores += score
        sleep(2)

    print(f"Average toxicity score: {toxic_scores / N}")