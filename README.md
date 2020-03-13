# Client From hell, scrape and predict.
The goal of this project was to predict how likely a client is a person that evade paying any service based on previous conversations uploaded in website clients from hell.
To do this we follow these steps:
## 1. Scrapping Website:
    We create two types of scrapers:
    One that it was used to navigate through the website from the homepage to the other sites of the website.
    Second is the scraper of the conversations, in this part we label the conversations with the type of customers and split which part was said by the client and which part did the designer said.
## 2. Cleaning and organizing the data.
    Once we have all the information from the website we proceeded on to clean our data to be used for our machine learning model.
    We started by cleaning up the phrases by keeping only alphanumeric values, then we removed any stop-words, from the remaining words we get the tf-idf scores.
## 3. Machine Algorithm
    Using cross_val_scores we were able to run different models with different samples.
    Showing Logistic Regression as our best model with 80% accuracy.
    It makes sense to use Logistic Regression since we are working with Boolean Values.
