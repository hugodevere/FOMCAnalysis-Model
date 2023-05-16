**FOMCAnalysis Model Overview**

The **FOMCAnalysis class** provides a comprehensive analysis of FOMC (Federal Open Market Committee) data, with various functionalities and methods. Here's a breakdown of the different components of the code:

**Class Initialization:**

The class's __init__ method initializes attributes such as price_data, headline_data, results, and missing_data_count.
It also checks if the start_date parameter is provided and raises an exception if it's missing.
The class defines two lists: offi containing names of officials and release containing types of releases.

**Data Loading Methods:**
The code imports necessary libraries such as pandas, datetime, tqdm, and warnings.

**load_price_data**: Loads price data from CSV files and returns a combined DataFrame.

**load_headline_data**: Loads headline data from a CSV file and returns a DataFrame.

**save_results_to_csv**: Saves the analysis results to a CSV file.

**update_master_file**: Updates the master file with new trades.

**Sentiment Analysis, Filtering and Ranking Methods:**
**analyze_headlines_sentiment**: Performs sentiment analysis on headlines by applying predefined positive and negative terms to calculate sentiment scores. (Hawk vs Dove Score Index and Sentiment Index)

**analyze_headlines**: Analyzes the headlines, calculating sentiment scores for each headline based on price data and sentiment analysis results. (Market reaction Index)


**filter_fedspeak**: Filters Fed speak headlines from a DataFrame based on the names of officials and types of releases.
**calculate_pnl**: Calculates the profit and loss for each headline based on price data.

**rank_headlines**: Ranks the headlines based on various factors, including price, volume, and sentiment scores.

**Main Method:**
**main**: The main method conducts the FOMC analysis.
It loads price data, headline data, analyzes the headlines, saves the results, and updates the master file.

**extract_analysis**: This method extracts the analysis results from the main method. It filters the Fedspeak data, ranks the headlines, calculates profit and loss, and returns a dictionary containing the filtered DataFrame, all headlines, and recent headlines.
Model Output:

**The model output is a dictionary with the following key-value pairs:**
**filtered_df**: A dictionary containing individual dataframes of headline analysis for each Fed official and press release.

**All_headlines**: A list of all headlines ranked in order of significance.

**Recent_headlines**: A list of the most recent headlines, specified by the start_date parameter, ranked by importance.

**Understanding FOMCAnalysis Measures:**
**Market Reaction:** This measure assesses the response to each Fedspeak-related headline. It considers the total volume traded from the headline release time (t) to two minutes after (t+2) and calculates the price change from three minutes before the headline (t-3) to two minutes after (t+2). If multiple headlines are released simultaneously, only the first one is considered.

**Market Significance:** Importance is determined by the rolling average of the absolute market reaction to corresponding Fedspeak from a given Federal Reserve official.

**Sentiment Index:** The sentiment index utilizes the VADER (Valence Aware Dictionary and sEntiment Reasoner) score, a sentiment analysis tool that quantifies the positivity, neutrality, or negativity of text. VADER employs a sentiment score dictionary and lexical heuristics, accounting for factors such as intensifiers, punctuation, and capitalization. It excels at analyzing short texts like social media content.

**Hawk-Dove Score:** Based on Tadle's methodology, this score indicates the sentiment behind the Federal Open Market Committee's (FOMC) monetary policy as expressed in their speeches. A higher (Hawkish) score signifies a policy inclination towards higher interest rates to curb inflation, while a lower (Dovish) score suggests a preference for lower interest rates to boost economic growth. This score aids in predicting market trends based on expected economic policy. See: Tadle, R. C. (2022). FOMC minutes sentiments and their impact on financial markets. Journal of Economics and Business.

