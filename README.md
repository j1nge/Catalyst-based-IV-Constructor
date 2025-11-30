## Methodology

### **Data Collection and Preprocessing**
We chose ANET (Arista Networks) because of good liquidity and decently well-defined earnings events. We obtained its short-maturity options (7D, 14D, 21D, 30D) chains through yfinance

Cleaning:
- Remove stale quotes, bid/ask outliers, zero-volume strikes
- Compute mid-market IV
- Exclude dates near other events when establishing diffusion IV (earnings +- 2days) baseline

### **Construct Diffusive Volatility Curve (Baseline IV)**
Identify set of "quiet" dates:
- 20-60 trading days before the event
- exclude event week

Fitting the baseline curve:
- Collect all ATM IV observations during through quiet windows
- Estimate diffusive IV using exponential fit:
![image.png](attachment:image.png)

### **Extract Event-Based Volatility (Event IV)**
- Take IV for maturity T that contain T days of variance. ie Total variance = norma variance + event variance
![image-2.png](attachment:image-2.png)
- Subtract diffusion IV for same T and convert both to variance in order to get the vol that market priced for the earnings move
![image-3.png](attachment:image-3.png)

Do this for 7d, 14d, 21d, 30d and average them to get one event IV number

### **Residual Volatility**
Event vol should fall over tme after earnings but not necessarily back to baseline IV

Create simple model for now by pulling last 8-12 earnings events for ANET. 

1. Look at past earnings events and ather the last several ANET earnings cycles and pull the 30D (or chosen tenor) IV around each one.

2. Estimate the pre-event diffusion level and for each past event, use quiet days before earnings to estimate what “normal” IV looked like.

3. Measure the post-event IV level and look at IV on days +3 to +7 after the event, when the volatility spike has mostly settled.
Average these days to get the “steady” post-event level for that cycle.

4. Compute the residual shift and for each cycle, subtract the pre-event diffusion level from the steady post-event IV.
This tells you how much IV typically stays elevated after the earnings effect fades.

5. Average the residual shifts. Take the average across all historical earnings cycles since this gives the typical post-event elevation ANET experiences

6. Predict the new post-event IV level by taking today’s diffusion baseline and add the historical average shift. This should give a realistic forecast of where IV should settle after the upcoming event.



