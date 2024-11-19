![image](https://github.com/user-attachments/assets/db6e81da-9f89-4ec3-b792-ee58894b95e8)

## About the Project
This repository contains resources and code for our work on layoff prediction, integrating technical and textual data for more accurate predictions.

## Abstract
In recent years, many companies have announced layoffs, making it increasingly important to develop tools that can analyze and predict such events. Initially, we aimed to address the binary classification problem of determining whether a layoff will occur or not. However, this proved to be a challenging task due to the limited data available and difficulties in establishing a reliable baseline for comparison.

We shifted our attention to processing and integrating news data alongside technical financial metrics to tackle the percentage prediction task (i.e., predicting the percentage of employees being laid off, given that a layoff has occurred). Addressing the challenge of managing large volumes of news data, we experimented with filtering techniques, time window selection (7, 15, 30, and 90 days), and variations of FinBERT embeddings.

Our findings show that a hybrid approach combining technical and textual data can improve prediction accuracy. The best-performing model, which used average embeddings of unfiltered summaries, achieved a Test Mean Absolute Error (MAE) of 8.74, outperforming other configurations. This highlights the importance of retaining broader contextual information and suggests that advanced methods for integrating diverse data sources hold promise for further improving layoff prediction models.

## Paper
You can view the full paper [here](ProjectPaper.pdf).
