# Introduction
This project demonstrates the application of the K-means clustering algorithm on the KDD'99 dataset for anomaly detection. The process involves downloading the dataset (Task 1), performing K-means clustering (Task 2), and then evaluating the results using various metrics (Task 3). This repository provides the necessary scripts and documentation to reproduce the analysis.
All tasks were solved in a Kali Linux VM under Virtual Box.

## Task 1
Dataset Download in Kali Linux (Terminal): 
```
wget https://raw.githubusercontent.com/PacktWorkshops/The-Data-Science-Workshop/master/Chapter09/Dataset/KDDCup99.csv
```

## Task 2
### Prerequisites
- Python 3.x
- Kali Linux Virtual environment (optional but recommended)

### Installing Dependencies
1. Clone the repository.
2. Navigate to the project's root directory.
3. Install the required Python packages using:
    ```bash
    pip install -r requirements.txt
    ```
### Directory Structure
- [README.md](./README.md/): Contains the main documentation for the project.
- [data/raw](./data/raw/): This directory holds the raw data files. In this case, it contains the `KDDCup99.csv` dataset. We are working with this dataset.
- [results](./results/): This directory stores the output files generated by the analysis. Here, it includes `anomalies.csv` which lists the detected anomalies. "elbow.png" provides a visual representation of the optimal k. 'preData.png' and 'postData.png' represent the raw data without clustering and the clustered data after the kmeans algorithm.
- [src](./src/): Contains the source code for the project. The `kdd99_analysis.py` script performs the whole analysis.
- [requirements.txt](./requirements.txt/): Lists all Python packages installed in the execution environment. Contains all packages that are required to execute the code. This file ensures that all dependencies can be easily installed.

### Running the Code
1. Ensure that the `KDDCup99.csv` file is located in the `data/raw` directory.
2. Navigate to the `src` directory:
    ```bash
    cd src
    ```
3. Run the Python script:
    ```bash
    python3 kdd99_analysis.py
    ```


### Explanation of the K-Means Analysis and Determining the Number of Clusters
1. **Load Dataset**:
    - The dataset is loaded from the `data/raw/KDDCup99.csv` file.

2. **Data Preparation**:
    - String data is converted to numeric values since the K-Means algorithm requires numeric input.
    - Features are scaled to ensure they contribute equally to the analysis.

3. **Determining the Optimal Number of Clusters**:
    - The `KElbowVisualizer` is used to determine the optimal number of clusters. This visualizer plots the explained variance as a function of the number of clusters and looks for a "kink" in the curve, indicating the optimal number of clusters.
    - The optimal number of clusters is determined to be `8` as shown in the Terminal output.

4. **Performing K-Means Clustering**:
    - K-Means clustering is performed with the optimal number of `8` clusters.
    - Data points are assigned to clusters, and distances to the respective cluster centers are calculated.

5. **Anomaly Detection**:
    - Anomalies are defined as data points whose distance to the cluster centers exceeds the 95th percentile.
    - These anomalies are identified and saved to the `results/anomalies.csv` file.

6. **Results**:
    - The number of detected anomalies is printed to the terminal (`24701`).
    - The anomalies are saved in `results/anomalies.csv`.

### Conclusion
This project successfully applied the K-Means algorithm for anomaly detection on the KDD99 dataset. The optimal number of clusters was determined using the Elbow method, and anomalies were identified and saved accordingly.

## Task 3
### Prerequisites
- Python 3.x
- Kali Linux Virtual environment (optional but recommended)

### Running the Code
1. Navigate to the `src` directory:
    ```bash
    cd src
    ```
2. Run the Python script:
    ```bash
    python3 metrics_calculation.py
    ```

### Explanation of the Code
This Python script calcualtes metrics form K-means clustering algorithm applied to the KDD'99 dataset.

1. Execute [kdd99_analysis.py](./src/kdd99_analysis.py/)
- The script begins by executing 'kdd99_analysis.py' which performs K-means clustering on the KDD'99 dataset (from Task 2)
- This is done to get the neceessary output which we then can use to calculate the metrics.  

2. Load Detected Anomalies:
- The anomalies detected by the 'kdd99_analysis.py' script are then loaded from the corresponding file
- This file contains the data points identified as anomalies by the clustering algorithm.

3. Load Original Dataset:
- The original KDD'99 dataset is then loaded
- This dataset contains all the data points along with their true labels

4. Extract and Convert True Labels:
- The true labels from the dataset are extracted and converted into binary values
- 'Normal' instances are labeled as 0, and all other instances (attacks) are labeled as 1
- This binary representation is necessary for calculating the evaluation metrics.

5. Predict Anomaly Labels:
- An array of predicted labels is created, initially assuming all data points are normal (0)
- The indices of the data points identified as anomalies are updated to 1
- This array represents the clustering algorithm's predictions

6. Determination of TP, FP, TN, FN
- The script calculates the four components true positives (TP), false positives (FP), true negatives (TN), and false negatives (FN)
- These values are derived by comparing the predicted labels with the true binary labels.

7. Compute Evaluation Metrics:
- Various evaluation metrics are calculated based on the confusion matrix components: Precision, Recall, False Positive Rate (FPR), Specificity, Negative Predictive Value (NPV), Accuracy and F-Measure 

8. Store Metrics in DataFrame:
- The calculated metrics are stored in a pandas DataFrame

9. Generate PDF Report:
- The calculated metrics are then stored in a PDF file as a table (metrics_calculation.pdf](./results/metrics_calculation.pdf/))


### Evaluation for KDD'99 and K-means
This report presents the evaluation of the K-means clustering algorithm applied to the KDD'99 dataset using the metrics discussed in the lecture.


The following metrics were determined at first:
- True Positives (TP): Events correctly identified as attacks. These are events that were saved in [anomalies.csv](./results/anomalies.csv/) and were not labeled as “normal”
- False Positives (FP): Normal events incorrectly identified as attacks. These are events that were saved in [anomalies.csv](./results/anomalies.csv/) but labeled as “normal”
- True Negatives (TN): Normal events correctly identified as normal. These are events from `data/raw/KDDCup99.csv` that were not saved in the [anomalies.csv](./results/anomalies.csv/) and were labeled as “normal”
- False Negatives (FN): Attacks incorrectly identified as normal. These are events from `data/raw/KDDCup99.csv` that were not saved in [anomalies.csv](./results/anomalies.csv/) but were not labeled as “normal”

From the above basic values, the following derived metrics were calculated:


#### Precision
The ratio of correctly identified attacks to the total number of events identified as attacks.
- Precision = TP / TP+FP

#### Recall
The ratio of correctly identified attacks to the total number of actual attacks.
- Recall = TP / TP+FN

#### False Positive Rate (FPR)
The ratio of incorrectly identified attacks to the total number of actual normal events.
- FPR = FP / FP+TN

#### Specificity
The ratio of correctly identified normal events to the total number of actual normal events.
- Specificity = TN / FP+TN

#### Negative Predictive Value (NPV)
The ratio of correctly identified normal events to the total number of events identified as normal.
- NPV = TN / TN+FN

#### Accuracy
The ratio of correctly identified events (both normal and attacks) to the total number of events.
- Accurancy = TP+TN / TP+FP+TN+FN

#### F-Measure
The harmonic mean of precision and recall.
- F-Measure = (2xPrecisionxRecall)/(Precision+Recall)

#### Results
The metrics calculated for the K-means clustering algorithm on the KDD'99 dataset are as follows:

| Metric                   | Value        |
|--------------------------|--------------|
| True Positives (TP)      | 3890         |
| False Positives (FP)     | 20811        |
| True Negatives (TN)      | 76466        |
| False Negatives (FN)     | 392853       |
| Precision                | 0.1575       |
| Recall                   | 0.0098       |
| False Positive Rate (FPR)| 0.2139       |
| Specificity              | 0.7861       |
| Negative Predictive Value (NPV) | 0.1629 |
| Accuracy                 | 0.1627       |
| F-Measure                | 0.0185       |

These results were generated using the Python script [metrics_calculation.py](./src/metrics_calculation.py).

#### Discussion
The results of the K-means clustering analysis on the KDD'99 dataset show challenges and limitations.

1. Low Precision and low Recall:
The precision of 0.1575 and the recall of 0.0098 indicate a bad performance of the model. Low precision means that a high proportion of data points identified as anomalies are actually normal data (many false positives).
A low recall indicates that many anomalies are not detected (many false negatives). These results show that the K-means algorithm has difficulties in identifying anomalies.

2. High false positive rate (FPR): 
The FPR of 0.2139 is relatively high. Approximately 21.39% of normal data was incorrectly classified as anomalies. This leads to a high number of false alarms, which can increase the working effort and decrease the confidence in the system.

3. Specificity and NPV:
A specificity of 0.7861 and a negative predictive value (NPV) of 0.1629 show that the model is relatively good at recognizing normal events as such. But it has difficulty correctly identifying anomalies. The high number of false negatives (392,853) further show this problem.

4. Accuracy
The accuracy of 0.1627 shows that the model only made about 16.27% of the total predictions correctly. This is a further indication of the lack of ability of the K-means algorithm to be useful in this context.

5. F-Measure:
With an F-measure of 0.0185, we can see that the model performs not very good in terms of precision and recall. 

#### Performance Metrics
Additionally, the following performance metrics were considered but were deemed less relevant due to their dependency on the specific system and environment in which the script is executed:

- Memory Usage
- Packets or Log Events per Second
- Time to Train the Model
- Time to Perform Detection
- Computational Complexity

#### Conclusion
The evaluation of the K-means clustering algorithm on the KDD'99 dataset shows significant challenges. The low precision (0.1575) and recall (0.0098) show the issue in correctly identifying anomalies and minimizing false alarms. The high false positive rate (0.2139) shows also the issue of false alarms, which can burden the system and decrease confidence in the detection mechanism.
The accuracy of 0.1627 shows that the model is only slightly better than random guessing. The low F-measure (0.0185) shows the poor balance between precision and recall. The specificity (0.7861) suggests that while the model can identify normal instances correctly, it struggles significantly with anomalies, reflected in the high number of false negatives (392,853).

Additionally, it is important to note that the KDD'99 dataset itself has been criticized for being outdated and not fully representative of modern network traffic, containing redundant and irrelevant features. The dataset has a disproportionate distribution of attack types​ [1]​. These issues further complicate the evaluation and effectiveness of machine learning models trained on this dataset.


### References
[1] [Tavallaee, M., Bagheri, E., Lu, W., & Ghorbani, A. A. (2009, July). A detailed analysis of the KDD CUP 99 data set. In 2009 IEEE symposium on computational intelligence for security and defense applications (pp. 1-6). Ieee.](./https://www.ecb.torontomu.ca/~bagheri/papers/cisda.pdf#:~:text=URL%3A%20https%3A%2F%2Fwww.ecb.torontomu.ca%2F~bagheri%2Fpapers%2Fcisda.pdf%0AVisible%3A%200%25%20/)
