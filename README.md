# FOMCAnalysis

The FOMCAnalysis class divides into three parts: 

Data loading in which 





Model computation:



Data updating:



Understanding the FOMCAnalysis Measures:

Market Reaction: This is calculated by assessing the response to each Fedspeak-related headline. Specifically, we tally the total volume from the time of the headline (t) to two minutes after (t+2), and the price change from three minutes before the headline (t-3) to two minutes after (t+2). If multiple headlines are released at the same time, we only consider the first one.

Importance: This is gauged by the rolling average of the absolute market reaction to corresponding Fedspeak from a given Federal Reserve official.

Sentiment Index: This employs the VADER (Valence Aware Dictionary and sEntiment Reasoner) score, a sentiment analysis tool that measures the degree of positivity, neutrality, or negativity in text. The tool uses a combination of a sentiment score dictionary and lexical heuristics, taking into account factors such as intensifiers, punctuation, and capitalization. This makes it particularly adept at analyzing social media content and other short texts.

Hawk-Dove Score: Based on Tadle's methodology, this score indicates the sentiment behind the Federal Open Market Committee's (FOMC) monetary policy as expressed in their speeches. A higher (Hawkish) score signifies a policy inclination towards higher interest rates to curb inflation, while a lower (Dovish) score suggests a preference for lower interest rates to boost economic growth. This score aids in predicting market trends based on expected economic policy. See: Tadle, R. C. (2022). FOMC minutes sentiments and their impact on financial markets. Journal of Economics and Business.

