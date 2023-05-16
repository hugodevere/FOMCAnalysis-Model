class FOMCAnalysis:
    def __init__(self,restart_data=False,start_date=None):
        self.price_data = None
        self.headline_data = None
        self.results = None
        self.missing_data_count = 0
        self.restart_data=restart_data
        if start_date is None:
            raise Exception('You must specify start_date of the analysis!')
        else:
            self.start_date= start_date

        self.offi = ['Collins','Evans','Daly','Harker','Bostic','Williams','Cook','Brainard','Barr','Jefferson','Bowman','Powell','Waller','Barkin','Logan','Kaskari','George','Mester','Bullard']
        self.release = ['Beige','Minutes']

    import pandas as pd
    from datetime import datetime
    from tqdm.auto import tqdm
    import warnings
    warnings.filterwarnings('ignore')

    def load_price_data(self, filenames, base_path):
        """
        This function loads price data from csv files.
        :param filenames: A list of csv files to load.
        :param base_path: The base path where the csv files are located.
        :return: A combined dataframe of all the csv files.
        """
        from tqdm.auto import tqdm
        import pandas as pd
        dfs = []
        for filename in tqdm(filenames, desc='Extracting price data'):
            df = pd.read_csv(base_path + filename, index_col='Date')
            df.index = pd.to_datetime(df.index)
            df.index.strftime('%d/%m/%y %H:%M:%S')
            df = df.tz_localize('GMT')
            df = df.tz_convert('EST')
            if filename == '\SFR3M_recent.csv':
                df = df[df.index.year > 2022]
            df.columns = ['Price', 'Volm']
            print(f'{filename} start from {df.head(1).index[0]} to {df.tail(1).index[0]}')
            dfs.append(df)

        combined_df = pd.concat(dfs, ignore_index=False)
        combined_df = combined_df.drop_duplicates(keep='first')
        combined_df = combined_df.sort_index(ascending=True)
        return combined_df


    def load_headline_data(self, file_path):
        """
        This function loads headline data from a csv file.
        :param file_path: The path of the csv file.
        :return: A dataframe of the csv file.
        """
        import pandas as pd
        try:
            # Try to read the file with UTF-8 encoding
            df = pd.read_csv(file_path, index_col='Date')
        except UnicodeDecodeError:
            # If that fails, try with "ISO-8859-1" encoding
            df = pd.read_csv(file_path, index_col='Date', encoding='ISO-8859-1')
        df.index = pd.to_datetime(df.index)
        df.index.strftime('%d/%m/%y %H:%M')
        df = df.sort_index(ascending=True)
        df = df.tz_convert('EST')
        df = df[~df.index.duplicated(keep='first')].sort_index(ascending=True)
        return df


    def save_results_to_csv(self, results, file_path):
        """
        This function saves the results to a csv file.
        :param results: A dataframe of results.
        :param file_path: The path where the csv file will be saved.
        :return: The results dataframe.
        """
        results.to_csv(file_path)
        return results

    def update_master_file(self, new_trades, master_file_path):
        """
        This function updates the master file with new trades.
        :param new_trades: A dataframe of new trades.
        :param master_file_path: The path of the master file.
        :return: A dataframe of the updated master file.
        """
        import pandas as pd
        all_data = pd.read_csv(master_file_path, index_col='Date')
        all_data.index = pd.to_datetime(all_data.index)
        all_data = all_data.tz_convert('EST')
        all_data = new_trades.append(all_data).sort_index(ascending=True)
        all_data = all_data[~all_data.index.duplicated(keep='first')]
        all_data.to_csv(master_file_path)
        return all_data
    
    def analyze_headlines_sentiment(self, headlines):
        """
        This function analyses the sentiment of headlines.
        :param headlines: A list of headlines.
        :return: A list of sentiment scores.
        Please note that the concept of sentiment analysis and the methodology used in this function are inspired by the work of Tadle, R. C. (2022)
        in their publication titled "FOMC minutes sentiments and their impact on financial markets" published in the Journal of Economics and Business.
        """
        positive_terms = ['abatinga', 'accelerated', 'add' ,'advance' ,'advanced', 'augmented', 'balanced' , 'better', 'bolsters', 'boom', 'booming', 'boost', 'boosted' ,'eased', 'elevated',
                      'elevating', 'expand expanding', 'expansionary', 'extend', 'extended', 'fast', 'faster', 'firmer', 'gains', 'growing', 'heightened', 'high', 'higher', 'improved', 'improvement',
                      'improving', 'increase', 'increased','increases', 'increasing', 'more','raise','rapid','rebounded','recovering','rise','risen','rising','robust' 'rose',
                      'significant','solid','sooner','spike', 'spikes','spiking','stable','strength','strengthen','strengthened','strengthens','strong','stronger','supportive',
                      'up','upside','upswing', 'uptick']
    
        negative_terms = ['adverse','back','below','constrained','contract','contracting','contraction','cooling','correction','dampen','damping','decelerated','decline','declined','declines',
                'declining','decrease','decreases','decreasing','deepening', 'depressed','deteriorated','deterioration','diminished', 'disappointing', 'dislocation','disruptions','down', 
                'downbeat', 'downside', 'drop', 'dropping', 'ebbed', 'erosion', 'fade', 'faded', 'fading', 'fall', 'fallen', 'falling', 'fell', 'insufficient', 'less', 'limit', 'low', 'lower',
                    'moderated', 'moderating', 'moderation', 'reduce', 'reduced', 'reduction', 'reluctant', 'removed', 'restrain', 'restrained', 'restraining', 'restraint', 'resumption'
                    'reversed', 'slack', 'slow', 'slowed','slower','slowing' ,'slowly','sluggish', 'sluggishness','slumped','soft','softened','softening', 'stimulate', 'strained', 'strains',
                    'stress','subdued', 'tragic', 'turmoil', 'underutilization', 'volatile', 'vulnerable', 'wary', 'weak', 'weakened', 'weaker', 'weakness', 'a', 'The', 'term']
        hawkish_keys = ['business', 'businesses', 'demand', 'economic', 'economy', 'employment', 'energy', 'equities', 'equity', 'expansion', 'financial','growth','housing','income','indicators',
                    'inflation','inflationary','investment','investments','labor','manufacturing','outlook','output', 'price', 'prices', 'production', 'recovery', 'resource', 'securities',
                    'slack', 'spending', 'target', 'toll', 'wage', 'wages'] 
        dovish_keys = ['accommodation','devastation','downturn','recession','unemployment'] 

        def preprocess_headlines(headlines):
            import pandas as pd
            from tqdm.auto import tqdm
            from nltk.corpus import stopwords
            from nltk.tokenize import word_tokenize
            preprocessed_headlines = []
            for hl in tqdm(headlines, desc='pre-processing headlines'):
                words = word_tokenize(hl)
                words = [word.lower() for word in words if word.lower() not in stopwords.words('english')]
                preprocessed_headlines.append(words)
            return preprocessed_headlines
        
        def calculate_sentiment_scores(headlines):
            import nltk
            from nltk.sentiment import SentimentIntensityAnalyzer
            from tqdm import tqdm
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('vader_lexicon')
            import pandas as pd
            sia = SentimentIntensityAnalyzer()
            sentiment_scores = []
            for hl in tqdm(headlines, desc='Analyzing headlines sentiment'):
                vader_score = sia.polarity_scores(" ".join(hl))['compound']
                sentiment_score = 0
                if any(word in hl for word in negative_terms):
                    sentiment_score = 1
                elif any(word in hl for word in positive_terms):
                    sentiment_score = -1
                if any(word in hl for word in dovish_keys):
                    sentiment_score -= 1
                elif any(word in hl for word in hawkish_keys):
                    sentiment_score += 1
                combined_score = vader_score + sentiment_score
                sentiment_scores.append([sentiment_score, vader_score, combined_score])
            return sentiment_scores
        
        preprocessed_headlines = preprocess_headlines(headlines)
        combined_results = [[score[0], score[1]] for score in calculate_sentiment_scores(preprocessed_headlines)]
        return combined_results
        
    def analyze_headlines(self, price_data, headlines):
        """
        This function analyses the headlines and calculates sentiment scores.
        :param price_data: A dataframe of price data.
        :param headlines: A dataframe of headlines.
        :return: A tuple of results and count of missing data.
        """
        import pandas as pd
        from tqdm.auto import tqdm
        import datetime
        results = []
        missing_data_count = 0
        for i in tqdm(range(0,len(headlines.index)), desc='Analyzing headlines'):
            idx = headlines.index[i]
            rounded_idx = idx.round('min')
            hl = headlines['Text'][i]
            start_time = rounded_idx - pd.Timedelta(minutes=2)
            end_time = rounded_idx + pd.Timedelta(minutes=3)
            trade = price_data[(price_data.index >= start_time) & (price_data.index <= end_time)].sort_index(ascending=True)
            interval_min = datetime.timedelta(minutes=6)
            if (trade.tail(1).index - trade.head(1).index) <= interval_min:
                total = pd.DataFrame()
                total['Price'] = (trade['Price'].tail(1).values - trade['Price'].head(1).values)
                total['Volm'] = sum(trade['Volm'].tail(3).values)
                results.append((rounded_idx, total.values[0], hl))
            else:
                missing_data_count += 1

        trades = pd.DataFrame(results, columns=['Date', 'Price_Volume', 'Headlines'])
        trades.set_index('Date', inplace=True)
        trades[['Price', 'Volume']] = pd.DataFrame(trades['Price_Volume'].tolist(), index=trades.index)
        trades.drop(columns=['Price_Volume'], inplace=True)
        
        headlines = trades['Headlines'].values
        sentiment_results = self.analyze_headlines_sentiment(headlines)
        trades['Fedspeak_hawk_dove_score'] = [x[0] for x in sentiment_results]
        trades['Fedspeak_sentiment_score'] = [x[1] for x in sentiment_results]
        return trades, missing_data_count
    
    def filter_fedspeak(self,df):
        """
        This function filters Fed speak from a dataframe.
        :param df: A dataframe.
        :return: A tuple of merged dataframe and a dictionary of filtered dataframes.
        """
        import pandas as pd
        # Filter headlines containing the names
        filtered_dfs = {}
        for name in self.offi:
            off_upper = df[df['Headlines'].str.contains(name.upper())]
            off_lower = df[df['Headlines'].str.contains(name)]
            off = off_upper.append(off_lower)
            filtered_dfs[name] = off
        # Filter headlines containing Beige and Minutes
        for r in self.release:
            r_upper = df[df['Headlines'].str.contains(r.upper())]
            r_lower = df[df['Headlines'].str.contains(r)]
            r_df = r_upper.append(r_lower)
            filtered_dfs[r] = r_df
        # Merge all filtered dataframes
        merged_df = pd.concat(filtered_dfs.values(), ignore_index=False)
        return merged_df, filtered_dfs
    
    def calculate_pnl(self,df, price):
        """
        This function calculates the profit and loss of each headline up to date.
        :param df: A dataframe.
        :param price: A dataframe of price data.
        :return: A numpy array of profit and loss.
        """
        import pandas as pd
        df = df.sort_index(ascending=True)
        date = df.index
        def change(date):
            r = []
            for i in date:
                change = (price['Price'].tail(1).values) - ((price[price.index >= str(i)]['Price']).head(1).values)
                r.append(change)
                r = list(r)
                for row, n in zip(df.itertuples(), range(0, len(r), 1)):
                    if row.Price > 0:
                        r[n] = r[n] * (1)
                    else:
                        r[n] = r[n] * (-1)
            return pd.DataFrame(r)
        R = change(pd.to_datetime(date))
        pnl = round(R, 2).values
        return pnl
    
    def rank_headlines(self,df, start_date=None, top_n=None):
        """
        This function ranks the headlines.
        :param df: A dataframe.
        :param start_date: A start date for ranking.
        :param top_n: The top n headlines to rank.
        :return: A tuple of dataframes of all trades and top n trades
        """
        import pandas as pd
        from datetime import datetime
        if start_date is None:
            start_date = pd.Timestamp((datetime.today()-pd.Timedelta('30 days'))).strftime('%Y-%m-%d')
        if top_n is None:
            top_n = 20
        print(start_date)
        def zscore(df):
            df = (df - df.mean()) / df.std(ddof=0)
            return df
        def rank_trades(data, start_date, top_n):
            data.index = pd.to_datetime(data.index)
            data = data.sort_index(ascending=True)
            if start_date is None:
                pass
            else:
                data = data[data.index > pd.Timestamp(start_date).tz_localize('EST')].copy()
            for col in ['Price', 'Volume']:
                if col == 'Price':
                    data[col + '_rank'] = abs(data[col]).rank(method='max')
                else:
                    data[col + '_rank'] = data[col].rank(method='max')
            data['Rank'] = data.filter(regex='_rank').sum(axis=1)
            data = data[data.columns.drop(list(data.filter(regex='_rank')))]
            data['Rank'] = round(data['Rank'].rank(pct=True), 2)
            data = data.sort_values('Rank', ascending=False)
            data.index.name = 'date'
            if start_date is None:
                pass
            else:
                data = data.head(top_n)
            for col in ['Price', 'Volume']:
                data[col] = round((data[[col]].apply(zscore).astype(float)), 1)
            return data

        def merge_sentiments(row):
            if row['Fedspeak_hawk_dove_score'] == 1 and row['Fedspeak_sentiment_score'] > 0:
                return 1
            elif row['Fedspeak_hawk_dove_score'] == -1 and row['Fedspeak_sentiment_score'] < 0:
                return -1
            else:
                return 0
        import pandas as pd
        df['Final_sentiment'] = df.apply(merge_sentiments, axis=1)
        return rank_trades(df.copy(), None, top_n),rank_trades(df.copy(), start_date, top_n)
    

    
    def main(self):
        """
        This is the main method that conducts the FOMC Analysis.

        Args:
        No arguments are required

        Returns:
        all_headlines (DataFrame): A DataFrame containing all headlines from the FOMC analysis.
        """
        import pandas as pd
        import time

        # Record the start time of the analysis
        start = time.time()

        # List of price data files to be imported
        price_data_files = [
            '\SFRU3_recent.csv',
            '\SFR3M_recent.csv',
            '\SFR3M_old.csv',
            '\SFR3H_PT3.csv',
            '\SFR3H_PT4.csv',
            '\SFR3H.csv',
            '\SFR3H_PT2.csv']
        
        base_path = r'C:\Users\hugod\OneDrive\Desktop\RATES DATA PYTHON'

        # Load the price data
        price_data = self.load_price_data(price_data_files, base_path)

        # Path to the recent headlines data file
        headline_data_file = r'C:\Users\hugod\OneDrive\Documents\RRATES\RATES\TAPE READING\Trading FOMC meetings\recent_headlines.csv'

        # Load the recent headlines data
        headline_data = self.load_headline_data(headline_data_file)

        # Analyze the headlines
        results, missing_data_count = self.analyze_headlines(price_data, headline_data)
        print(f'We don\'t have data for {missing_data_count}')

        # Save the results to a new CSV file
        new_trades = self.save_results_to_csv(results, 'Headlines_Fedspeak_new.csv')

        master_file_path = r'C:\Users\hugod\OneDrive\Documents\RRATES\RATES\TAPE READING\Trading FOMC meetings\Headlines_Fedspeak_all.csv'

        # Check if the user wants to restart the data
        if self.restart_data==True:
            # Ask the user for confirmation before restarting the data
            user_input = input("Are you sure you want to restart? This will overwrite the existing master file! Enter 'yes' to confirm: ")
            if user_input.lower() == 'yes':
                new_trades = self.save_results_to_csv(results,master_file_path)
                all_headlines = self.update_master_file(new_trades,master_file_path)
            else:
                # If the user didn't confirm, raise an exception
                raise Exception('Operation cancelled by user')
        else:
            all_headlines = self.update_master_file(new_trades,master_file_path)
        end = time.time()
        print('Total time: ', (end - start) / 60)
        return all_headlines,price_data
    
    def extract_analysis(self):
        headline_data,price_data = self.main()
        filtered_data = self.filter_fedspeak(headline_data)
        all_ranked,recent_ranked = self.rank_headlines(filtered_data[0],start_date=self.start_date)
        recent_ranked['PnL'] = self.calculate_pnl(recent_ranked,price_data)
        analysis = {}
        analysis['Filtered_df'] = filtered_data[1]
        analysis['All_headlines'] = all_ranked
        analysis['Recent_headlines'] = recent_ranked
        return analysis
