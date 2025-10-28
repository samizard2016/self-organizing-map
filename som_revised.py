import numpy as np
import pandas as pd
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pylab as plb
import matplotlib.pyplot as plt
import logging


class SelfOrganizingMap:
    def __init__(self, x, y, input_len, data, sigma=1.0, learning_rate=0.5):
        try:
            self.som = MiniSom(x, y, input_len, sigma=sigma, learning_rate=learning_rate, random_seed=42)
            self.som.random_weights_init(data)
            logging.basicConfig(
                       format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
                        datefmt="%d-%m-%Y %H:%M:%S",
                        level=logging.INFO,
                        filename='deep_cluster.log'
                        )
            self.logger = logging.getLogger("deep_cluster")
            """DEBUG INFO WARNING ERROR CRITICAL"""
            self.logger.info("A new session of Deep Cluster just started")
        except Exception as err:
            self.logger.error(f"There has been a trouble in initiating Deep Cluster: {err}")
    
    def train(self, data, num_iterations):
        try:
            self.som.train_random(data, num_iterations)
            self.logger.info(f"The model (Deep Cluster) has been trained with default settings")
        except Exception as err:
            self.logger.error(f"There has been a trouble in the initial training of Deep Cluster: {err}")
        
    def retrain(self,data,x,y,input_len,learning_rate,num_iterations):
        try:
            self.som = MiniSom(x, y, input_len, sigma=1, learning_rate=learning_rate)
            self.som.random_weights_init(data)
            self.train(data,num_iterations=num_iterations)
            self.logger.info(f"Deep Cluster has been retrained with lr {learning_rate} and iterations {num_iterations}")
        except Exception as err:
            self.logger.error(f"Retraining of Deep Cluster has failed with lr {learning_rate} and iterations {num_iterations}")
    
    def find_fraudulent_responses(self, data, threshold=0.5):
        quantization_errors = np.array([self.som.quantization_error([d]) for d in data])
        mean_error = np.mean(quantization_errors)
        std_error = np.std(quantization_errors)
        fraudulent_responses = [i for i, error in enumerate(quantization_errors) if error > mean_error + threshold * std_error]
        
        # Additional check for responses with all the same digit
        for i, d in enumerate(data):
            if len(set(d)) == 1:
                fraudulent_responses.append(i)
        
        return fraudulent_responses
    def get_outliers_threshold(self,data, threshold=2.0):
        """
        Identify outliers in the data based on their distance from the BMU in the trained SOM.

        Parameters:
        som (MiniSom): A trained MiniSom instance.
        data (np.ndarray): The input data (n_samples, n_features).
        threshold (float): The threshold for identifying outliers. Data points with a quantization error
                        greater than this threshold are considered outliers. Default is 2.0.

        Returns:
        outliers (list): The indices of the outlier data points.
        quantization_errors (np.ndarray): The quantization error for each data point.
        """
        try:
            quantization_errors = []
            outliers = []

            for i, x in enumerate(data):
                # Find the BMU for the data point
                bmu = self.som.winner(x)
                bmu_weights = self.som.get_weights()[bmu]

                # Calculate the quantization error (distance between the data point and its BMU)
                error = np.linalg.norm(x - bmu_weights)
                quantization_errors.append(error)

                # Check if the error exceeds the threshold
                if error > threshold:
                    outliers.append(i)
            self.logger.info(f"List of outliers based off threshold({threshold}) has been generated")
            return outliers, np.array(quantization_errors)
        except Exception as err:
            self.logger.error(f"List of outliers based off threshold({threshold}) couldn't be generated: {err}")
            
    
    def get_outlier_3sigma(self,data):
        """
        Identify outliers in the data based on their distance from the BMU in the trained SOM.

        Parameters:
        som (MiniSom): A trained MiniSom instance.
        data (np.ndarray): The input data (n_samples, n_features).
        threshold (float): The threshold for identifying outliers. Data points with a quantization error
                        greater than this threshold are considered outliers. Default is 2.0.

        Returns:
        outliers (list): The indices of the outlier data points.
        quantization_errors (np.ndarray): The quantization error for each data point.
        """
        try:
            quantization_errors = []
            outliers = []

            for i, x in enumerate(data):
                # Find the BMU for the data point
                bmu = self.som.winner(x)
                bmu_weights = self.som.get_weights()[bmu]
                # Calculate the quantization error (distance between the data point and its BMU)
                error = np.linalg.norm(x - bmu_weights)
                quantization_errors.append(error)

            mean_error = np.mean(quantization_errors)
            std_error = np.std(quantization_errors)
            for i, error in enumerate(quantization_errors):
                if error > mean_error + 3 * std_error:
                    outliers.append(i) 
            self.logger.info(f"3-σ list of outliers has been generated successfully")
            return outliers, np.array(quantization_errors)
        except Exception as err:
            self.logger.error(f"3-σ list of outliers has not been generated successfully: {err}")
            
    def visualize_distance_map(self,som):
        """
        Visualize the distance map (U-Matrix) of a trained MiniSom SOM.

        Parameters:
        som (MiniSom): A trained MiniSom instance.
        """
        # Get the weights from the SOM
        weights = som.get_weights()

        # Calculate the U-Matrix (distance map)
        umatrix = np.zeros((som._weights.shape[0], som._weights.shape[1]))

        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                neighbors = []
                if i > 0:
                    neighbors.append(weights[i-1, j])  # Top neighbor
                if i < weights.shape[0] - 1:
                    neighbors.append(weights[i+1, j])  # Bottom neighbor
                if j > 0:
                    neighbors.append(weights[i, j-1])  # Left neighbor
                if j < weights.shape[1] - 1:
                    neighbors.append(weights[i, j+1])  # Right neighbor

                # Calculate the average distance to neighbors
                if neighbors:
                    umatrix[i, j] = np.linalg.norm(weights[i, j] - np.mean(neighbors, axis=0))

        # Plot the U-Matrix
        plt.figure(figsize=(8, 8))
        plt.pcolor(umatrix.T, cmap='bone_r')  # Transpose for correct orientation
        plt.colorbar()
        plt.title('Distance Map (U-Matrix)')
        plt.show()
    def u_matrix(self):
        try:
            # visualize the distance matrix
            plb.bone()
            plb.pcolor(self.som.distance_map().T)
            plb.colorbar()
            img_file = "som distance map.png"
            plb.savefig(img_file,bbox_inches='tight', pad_inches=0.0)
            # Close the plot
            plb.close()
            self.logger.info(f"U Matrix has been successfully built and saved")
            return img_file
        except Exception as err:
            self.logger.error(f"There has been a challenge in building the U Matrix: {err}")

  
    @staticmethod
    def prepare_data(data):
        try:
            data = pd.get_dummies(data,dtype=float)
            sc = MinMaxScaler()
            tdata = sc.fit_transform(data)            
            return data,tdata
        except Exception as err:
             return err
    @staticmethod
    def prepare_data_advanced(df,unique_threshold=10):
        """
        Advanced method to determine column types and prepare DataFrame for SOM
        with additional preprocessing steps. Treats numeric columns with few unique
        values as categorical.
        
        Parameters:
        df (pandas.DataFrame): Input DataFrame    
        unique_threshold (int): Max number of unique values for a numeric column to be considered categorical
        
        Returns:
        tuple: (data,transformed_data)
        """
        # Detect column types
        column_types = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                unique_values = df[col].nunique()
                if unique_values <= unique_threshold:
                    column_types[col] = 'categorical'
                else:
                    column_types[col] = 'numeric'
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                column_types[col] = 'datetime'
            elif pd.api.types.is_bool_dtype(df[col]):
                column_types[col] = 'boolean'
            else:  # Includes string/object types
                column_types[col] = 'categorical'
        
        # Create a copy of the DataFrame
        df_transformed = df.copy()
        
        # Handle missing values
        for col in df.columns:
            if df_transformed[col].isna().any():
                if column_types[col] == 'numeric':
                    df_transformed[col] = df_transformed[col].fillna(df_transformed[col].mean())
                elif column_types[col] in ['categorical', 'boolean']:
                    df_transformed[col] = df_transformed[col].fillna(df_transformed[col].mode()[0])
                elif column_types[col] == 'datetime':
                    df_transformed[col] = df_transformed[col].fillna(df_transformed[col].mode()[0])
        
        # Transform categorical and boolean columns (including numeric-as-categorical)
        label_encoders = {}
        for col in df.columns:
            if column_types[col] in ['categorical', 'boolean']:
                le = LabelEncoder()
                df_transformed[col] = le.fit_transform(df_transformed[col].astype(str))
                label_encoders[col] = le
        
        # Convert datetime to numeric (timestamp)
        for col in df.columns:
            if column_types[col] == 'datetime':
                df_transformed[col] = df_transformed[col].astype(np.int64) // 10**9
        
        # Convert to numpy array
        data = df_transformed.values
        
        # Scale the data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)  
        
        return df, data_scaled
if __name__=="__main__":
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

    dfx = pd.concat([df, df1], axis=0)
    data = pd.get_dummies(dfx,dtype==float)
    sc = MinMaxScaler()
    data = sc.fit_transform(data)

    # Add respondent IDs
    dfx['Respondent_ID'] = range(1, num_respondents + 11)

    # Set Respondent_ID as the index
    dfx.set_index('Respondent_ID', inplace=True)

    # Display the dataset
    print(dfx)

    som = SelfOrganizingMap(10, 10, dfx.shape[1], dfx.values)
    som.train(dfx.values, 200)
    fraudulent_responses = som.find_fraudulent_responses(dfx.values)
    print(f"Fraudulent responses: {fraudulent_responses}")
    print(dfx.loc[fraudulent_responses])