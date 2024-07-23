from datasets import load_dataset
from dotenv import load_dotenv
import os
from openai import OpenAI
import requests
import json
from time import sleep
from tqdm import tqdm

load_dotenv()

# Initialize the OpenAI client
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

def get_toxicity_score(text, max_tries=3):
    comment = {
        'comment': {'text': text},
        'languages': ['en'],
        'requestedAttributes': {'TOXICITY': {}}
    }
    url = (f'https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={os.environ.get("PERSPECTIVE_API_KEY")}')

    for attempt in range(max_tries):
        try:
            response = requests.post(url, data=json.dumps(comment))
            response = response.json()
            if 'attributeScores' in response and 'TOXICITY' in response['attributeScores']:
                score = response['attributeScores']['TOXICITY']['summaryScore']['value']
                return score
            else:
                print(f"Retrying... {1 + attempt}/{max_tries}")
                sleep(2)
        except Exception as e:
            print(f"Error: {e}")
            print(f"Retrying... {1 + attempt}/{max_tries}")
            sleep(2)

    return None


if __name__ == '__main__':

    # Hyperparameters
    N = 1000
    OPENAI_MODEL = "gpt-3.5-turbo-instruct"

    # Load the dataset
    dataset = load_dataset("allenai/real-toxicity-prompts")['train'][:N]['prompt']

    # Loop through the dataset
    toxic_scores = 0

    for prompt in tqdm(dataset):
        response = client.completions.create(
            model=OPENAI_MODEL,
            prompt=prompt['text']
        )
        response_text = response.choices[0].text

        # Calculate the toxicity score
        score = get_toxicity_score(response_text)
        toxic_scores += score
        sleep(2)

    print(f"Average toxicity score of model {OPENAI_MODEL} with {N} data is {toxic_scores / N}")