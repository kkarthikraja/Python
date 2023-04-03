import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("Bechdel Test.csv")
df.head()
df.info()

# Creating a column for total_score including the Bechdel, Waithe, and Ko tests
df["total_score"] = df['bechdel'] + df['waithe'] + df['ko']
df.head()

# Sorting Data
df_sorted = df.sort_values('total_score').reset_index(drop = True)
df_sorted.head()

# Isolating the Data
df_partial = df_sorted[['movie', 'bechdel', 'waithe', 'ko', 'total_score']]
df_partial.head()

# Plot DataFrame with Matplotlib
ax = df_partial[["movie", "total_score"]].set_index('movie')
ax.plot(kind='bar', title="Representation In Movies", figsize=(15,10), legend=True)

# Horizontal Bargraph
ax.plot(kind='barh', title ='Representation In Movies', figsize=(15, 15), legend=True, fontsize=12)
