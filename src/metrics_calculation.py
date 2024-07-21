import subprocess
import pandas as pd
import numpy as np
from fpdf import FPDF

# Execute the K-Means clustering script from Task 7
subprocess.run(['python3', 'kdd99_analysis.py'])

# Loading the anomalies from Task 7
anomalies = pd.read_csv('../results/anomalies.csv')

# Load the original data set
data = pd.read_csv('../data/raw/KDDCup99.csv')

# The original labels
true_labels = data['label'].values

# Convert the true labels to binary values (1 = anomaly, 0 = normal)
true_labels_binary = np.where(true_labels == 'normal', 0, 1)

# The predicted labels (1 = anomaly, 0 = normal)
predicted_labels = np.zeros(len(data))
predicted_labels[data.index.isin(anomalies.index)] = 1

# Calculation of true positives, false positives, true negatives and false negatives
tp = np.sum((predicted_labels == 1) & (true_labels_binary == 1))
fp = np.sum((predicted_labels == 1) & (true_labels_binary == 0))
tn = np.sum((predicted_labels == 0) & (true_labels_binary == 0))
fn = np.sum((predicted_labels == 0) & (true_labels_binary == 1))

# Calculate metrics
precision = tp / (tp + fp) if (tp + fp) != 0 else 0
recall = tp / (tp + fn) if (tp + fn) != 0 else 0
false_positive_rate = fp / (fp + tn) if (fp + tn) != 0 else 0
specificity = tn / (fp + tn) if (fp + tn) != 0 else 0
npv = tn / (tn + fn) if (tn + fn) != 0 else 0
accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) != 0 else 0
f_measure = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

# Save results in a DataFrame
results = pd.DataFrame({
    "Metric": ["True Positives (TP)", "False Positives (FP)", "True Negatives (TN)", "False Negatives (FN)",
               "Precision", "Recall", "False Positive Rate", "Specificity", "Negative Predictive Value",
               "Accuracy", "F-Measure"],
    "Value": [tp, fp, tn, fn, precision, recall, false_positive_rate, specificity, npv, accuracy, f_measure]
})

# Save table with calculated values as a PDF file
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Evaluation Metrics', 0, 1, 'C')

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, 'Page %s' % self.page_no(), 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(10)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()

    def add_table(self, data):
        self.set_font('Arial', '', 12)
        col_width = self.w / 2.5
        row_height = self.font_size * 1.5
        for row in data.iterrows():
            self.cell(col_width, row_height, str(row[1]['Metric']), border=1)
            self.cell(col_width, row_height, str(row[1]['Value']), border=1)
            self.ln(row_height)

pdf = PDF()
pdf.add_page()
pdf.chapter_title('KDD\'99 Evaluation Metrics')
pdf.add_table(results)
pdf_output_path = '../results/metrics_calculation.pdf'
pdf.output(pdf_output_path)

print(f"Evaluation metrics have been saved to {pdf_output_path}")
