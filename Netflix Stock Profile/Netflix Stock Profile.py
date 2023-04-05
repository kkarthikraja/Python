from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

# Reading CSV Files
netflix_stocks = pd.read_csv("NFLX.csv")
print(netflix_stocks)

dowjones_stocks = pd.read_csv("DJI.csv")
print(dowjones_stocks)

netflix_stocks_quarterly = pd.read_csv("NFLX_daily_by_quarter.csv")
print(netflix_stocks_quarterly)

# Changing "Adj Close" to "Price"
dowjones_stocks.rename(columns={"Adj Close" : "Price"}, inplace=True)
netflix_stocks.rename(columns={"Adj Close" : "Price"}, inplace=True)
netflix_stocks_quarterly.rename(columns={"Adj Close" : "Price"}, inplace=True)

# Violin Plot
ax = sns.violinplot()
sns.violinplot(data=netflix_stocks_quarterly, x="Quarter", y="Price")

ax.set_title("Distribution of 2017 Netflix Stock Prices by Quarter")
ax.set_ylabel("Closing Stock Price")
ax.set_xlabel("Business Quarters in 2017")
plt.show()

# EPS 
x_positions = [1, 2, 3, 4]
chart_labels = ["1Q2017","2Q2017","3Q2017","4Q2017"]
earnings_actual =[.4, .15,.29,.41]
earnings_estimate = [.37,.15,.32,.41 ]

plt.scatter(x_positions, earnings_actual, color="red", alpha=0.5)
plt.scatter(x_positions, earnings_estimate, color="blue", alpha=0.5)
plt.legend(["Actual", "Estimate"])
plt.xticks(x_positions, chart_labels)
plt.title("Earnings Per Share in Cents")

# Side-by-Side Barchart
# The metrics below are in billions of dollars
revenue_by_quarter = [2.79, 2.98,3.29,3.7]
earnings_by_quarter = [.0656,.12959,.18552,.29012]
quarter_labels = ["2Q2017","3Q2017","4Q2017", "1Q2018"]

# Revenue
n = 1  # This is our first dataset (out of 2)
t = 2 # Number of dataset
d = len(revenue_by_quarter) # Number of sets of bars
w = .8 # Width of each bar
bars1_x = [t*element + w*n for element
             in range(d)]
plt.bar(bars1_x, revenue_by_quarter)

# Earnings
n = 2  # This is our second dataset (out of 2)
t = 2 # Number of dataset
d = len(earnings_by_quarter) # Number of sets of bars
w = .8 # Width of each bar
bars2_x = [t*element + w*n for element
             in range(d)]
plt.bar(bars2_x, earnings_by_quarter)


middle_x = [ (a + b) / 2.0 for a, b in zip(bars1_x, bars2_x)]
labels = ["Revenue", "Earnings"]

plt.title("Netflix Revenue vs. Earnings")
plt.xticks(middle_x, quarter_labels)
plt.legend(labels)

# Plotting Netflix and Dow Jones
# Left plot Netflix
# ax1 = plt.subplot(total number rows, total number columns, index of subplot to modify)
ax1 = plt.subplot(1,2,1)
plt.plot(netflix_stocks.Date, netflix_stocks.Price)
ax1.set_title("Netflix")
ax1.set_xlabel("Date")
ax1.set_ylabel("Stock Price")


# Right plot Dow Jones
# ax2 = plt.subplot(total number rows, total number columns, index of subplot to modify)
ax2 = plt.subplot(1,2,2)
plt.plot(dowjones_stocks.Date, dowjones_stocks.Price)
ax2.set_title("Dow Jones")

plt.subplots_adjust(wspace=.5)
plt.show()