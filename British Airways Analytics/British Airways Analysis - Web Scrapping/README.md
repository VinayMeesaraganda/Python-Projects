# British Airways Airline Analysis using Web Scrapping

This project performs web scraping on the [Airline Quality Reviews](https://www.airlinequality.com/airline-reviews/british-airways/) website to extract customer reviews for British Airways. The data is analyzed to gain insights into different aspects of British Airways' service. 

## Data Collection

- The Airline Quality Reviews website is scraped using the Requests and Beautiful Soup libraries in Python to extract customer reviews and ratings data.

- Reviews and ratings across different pages are extracted by looping through the paginated results. 

- The following fields are extracted for each review:

  - Aircraft
  - Type of Traveler
  - Seat Type 
  - Route
  - Date Flown
  - Ratings for:
    - Seat Comfort
    - Cabin Staff Service
    - Food & Beverages 
    - Inflight Entertainment
    - Ground Service
  - Value for Money
  - Recommendation
  - Review text

## Data Cleaning
  
- Duplicate reviews are dropped
- Missing values are filled in with appropriate averages  
- Irrelevant columns like Aircraft and Inflight Entertainment that have a lot of missing values are dropped

## Analysis

- Sentiment analysis is performed on the review text to classify each review as Positive, Negative or Neutral

- Summary statistics like review volume, ratings and sentiment distribution are visualized

- Text analysis techniques like word clouds and frequency plots are used to highlight most common words 

- Category rating is compared across different aspects like Seat Comfort, Staff Service etc.

## Conclusion

This analysis provides a comprehensive view into British Airways' service and performance based on real customer reviews. The techniques used can be extended to perform similar analysis for any airline.
