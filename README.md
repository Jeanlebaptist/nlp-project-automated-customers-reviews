README

Project NLP | Business Case: Automated Customer Reviews

Project Goal
This project aims to develop an automated product review analysis system powered by NLP models that aggregate customer feedback from a dataset. The key tasks include classifying review sentiment, clustering reviews to identify themes, and using generative AI to summarize insights.

Problem Statement
With thousands of reviews available across multiple platforms, manually analyzing them is inefficient. This project seeks to automate the process using an NLP pipeline to extract insights and provide a system for real-time analysis of customer feedback.

Main Tasks
1. Review Classification
Objective: Classify customer reviews into positive, negative, or neutral categories to help the company gain insights into product and service feedback.

Methodology:

Data Preparation: The project began by addressing a critical data imbalance issue using RandomOverSampler to create a more robust training dataset. The data was also preprocessed and cleaned to prepare it for the sentiment model.

Model Building: A pre-trained distilbert-base-uncased model was fine-tuned for sentiment classification, leveraging its powerful language representations to perform the task without training from scratch.

Model Evaluation: The model's performance was evaluated on a test dataset to ensure its effectiveness.

2. Customer Review Clustering
Objective: Simplify the dataset by clustering all review texts into distinct, broader categories.

Methodology:

An unsupervised learning approach was used to group reviews based on content similarity. This enabled the identification of recurring themes and topics within the customer feedback without the need for manual labeling.

3. Review Summarization Using Generative AI
Objective: Summarize reviews to generate insights into key product characteristics, complaints, and recommendations.

Methodology:

The project uses a zero-shot classification approach to categorize review texts.

A generative AI summarization model was then applied to create concise summaries for each category. This allows for quick, high-level insights into the key takeaways from large volumes of reviews.

Deployment Guidelines
The project includes a live web application that showcases the functionality of the three main components.

Framework: The web application was developed and deployed using the Gradio framework.

Functionality: The deployed app allows a user to input a new review, which is then processed by the sentiment classifier, the clustering model, and the summarization component. The results are displayed in real-time, demonstrating the full capability of the NLP pipeline.