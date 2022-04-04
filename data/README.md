# Datasets

We use the following datasets for our experiments: 

* The **Amazon Reviews** dataset consists of 21,000 English Amazon Reviews labeled as either real or fake. The reviews cover a variety of products with no particular product dominating the dataset. We filter out the non-english reviews and mark the fake and real reviews as deceptive and non-deceptive respectively. The final dataset is evenly balanced with 10,493 deceptive samples and 10,481 non-deceptive samples. 

* The Employment Scam Aegean Dataset, henceforth referred to as the **Job Scams** dataset consists of 17,880 human-annotated job listings. We use the description field as the sample text and the fraudulent field as our label. We clean the descriptions by removing all HTML tags, empty descriptions, and duplicates. Our final dataset is a heavily unbalanced dataset with 603 deceptive and 14,173 non-deceptive samples.

* The **Email Benchmarking dataset** is a phishing dataset consisting of both legitimate and phishing emails. We use the bodies extracted using PhishBench as the text.

* The **Liar** dataset consists of political statements made by US speakers assigned a fine-grain truthfullness label by PolitiFact. We use the claim as the text, and label  "pants-fire," "false," "barely-true" "half-true," as deceptive and "mostly-true," and "true" claims as non-deceptive.
  * It contains 5669 deceptive and 7167 truthful statements. 
  * On 11/27/2021, we noticed that a lot of the sample claims start with the phrase "Says that [claim]" or "Says [claim]". We modified our cleaning script to remove this signature. 

* The **WELFake** dataset is a more general fake news dataset intended to prevent overfitting. It combines 72,134 news articles from four pre-existing datasets (Kaggle, McIntire, Reuters, and BuzzFeed Political). It is roughly balanced with 35,028 real news articles and 37,106 fake news articles. As with the LIAR dataset, we consider fake news as deceptive and real news as truthful. 

