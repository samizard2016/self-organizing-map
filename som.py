"""
A self-organizing map (SOM) is an unsupervised machine learning technique used to produce a
low-dimensional (typically two-dimensional) representation of a higher-dimensional data set 
while preserving the topological structure of the data. It is a type of artificial neural network
that is trained using competitive learning rather than the error-correction learning used by other
neural networks. SOMs are useful for clustering and visualizing high-dimensional data.

To use SOMs for detecting fraudulent responses in research surveys, we can cluster the survey
responses and identify outliers or anomalies. Fraudulent responses are likely to be significantly
different from genuine responses and will appear as outliers in the SOM.

This class initializes a SOM, trains it with the provided data, and identifies fraudulent responses
based on their distance from the mean distance of the clusters. We can adjust the threshold parameter
to control the sensitivity of the fraud detection.

1: [Self-organizing map - Wikipedia](https://en.wikipedia.org/wiki/Self-organizing_map)
2: [Self-Organizing Maps: An Intuitive Guide with Python Examples](https://www.datacamp.com/tutorial/self-organizing-maps)


Here is a Python class to implement a self-organizing map using the MiniSom library:
"""

import numpy as np
import pandas as pd
from minisom import MiniSom

class SelfOrganizingMap:
    def __init__(self, x, y, input_len, data,sigma=1.0, learning_rate=0.5):
        self.som = MiniSom(x, y, input_len, sigma=sigma, learning_rate=learning_rate)
        self.som.random_weights_init(data)
    
    def train(self, data, num_iterations):
        self.som.train_random(data, num_iterations)
    
    def find_fraudulent_responses(self, data, threshold=0.5):
        distances = np.array([self.som.winner(d) for d in data])
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        fraudulent_responses = [i for i, d in enumerate(distances) if np.linalg.norm(d - mean_distance) > threshold * std_distance]
        return fraudulent_responses

# Example usage
# data = np.random.rand(100, 10)  # Replace with your survey data
# som = SelfOrganizingMap(10, 10, data.shape[1],data)
# som.train(data, 100)
# fraudulent_responses = som.find_fraudulent_responses(data)
# print(f"Fraudulent responses: {fraudulent_responses}")


# Number of respondents and questions
num_respondents = 100
num_questions = 10

# Generate random responses (choices between 1 and 5)
responses = np.random.randint(1, 6, size=(num_respondents, num_questions))
responses1 = [[5 for _ in range(10)] for s in range(10)]

# Create a DataFrame
columns = [f'Question_{i+1}' for i in range(num_questions)]
df = pd.DataFrame(responses, columns=columns)
df1 = pd.DataFrame(responses1, columns=columns)

dfx = pd.concat([df,df1],axis=0)

# Add respondent IDs
dfx['Respondent_ID'] = range(1, num_respondents + 11)

# Set Respondent_ID as the index
dfx.set_index('Respondent_ID', inplace=True)

# Display the dataset
print(dfx)

som = SelfOrganizingMap(10, 10, dfx.shape[1],df.values)
som.train(dfx.values, 200)
fraudulent_responses = som.find_fraudulent_responses(dfx.values)
print(f"Fraudulent responses: {fraudulent_responses}")
