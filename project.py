'''
Headspace: Analysis of the App’s Effectiveness

Steps:
    1. use app_store_scraper to download the most recent reviews
    2. scrape reviews and save it to csv
    3. clean the data
    4. perform sentiment analysis (TextBolb)
    5. normalize ratings --> compare stars (1-5) and sentiment (-1 to 1), normalize ratings to the same range
    6. visualize the data
    7. analyze correlation --> how well do ratings match sentiment?
    8. create bar graph of average sentiment by rating
    9. linear regression analysis on individual data points
    10. linear regression analysis on average sentiment values

'''
import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
from scipy import stats

# create sentiment analysis function
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity  # range: [-1.0, 1.0]


def main():
    # clean the data
    df = pd.read_csv("headspace.csv")
    df = df[["review", "rating", "date"]].dropna()
    df = df[df["review"].str.strip() != ""]

    # sentiment analysis
    # get a numeric sentiment score for each review
    df["sentiment"] = df["review"].apply(get_sentiment)

    # normalize ratings
    df["rating_normalized"] = (df["rating"] - 1) / 4

    # visualize data (scatterplot)
    plt.scatter(df["rating_normalized"], df["sentiment"], alpha=0.3)
    plt.xlabel("Normalized Star Rating")
    plt.ylabel("Sentiment Score")
    plt.title("Sentiment Score vs. Rating")
    plt.grid(True)
    plt.show()

    # analyze correlation
    correlation = df["rating_normalized"].corr(df["sentiment"])
    print("Correlation:", correlation)

    # calculate average sentiment for each star rating
    sentiment_by_rating = df.groupby("rating")["sentiment"].mean().reset_index()

    # Print the average sentiment for each rating
    print("Average Sentiment by Star Rating:")
    for _, row in sentiment_by_rating.iterrows():
        print(f"Rating {row['rating']}: {row['sentiment']:.4f}")

    # create bar graph of average sentiment by rating
    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        sentiment_by_rating["rating"],
        sentiment_by_rating["sentiment"],
        color='skyblue',
        width=0.6
    )

    # add data labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 0.01,
            f'{height:.3f}',
            ha='center',
            va='bottom',
            fontweight='bold'
        )

    plt.xlabel("Star Rating")
    plt.ylabel("Average Sentiment Score")
    plt.title("Average Sentiment Score by Star Rating")
    plt.xticks(sentiment_by_rating["rating"])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("sentiment_by_rating.png")
    plt.show()

    # linear regression analysis on individual data points
    # prepare data for regression
    x = df["rating"]
    y = df["sentiment"]

    # calculate the linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    # calculate correlation coefficient for individual data points
    correlation_individual = np.corrcoef(x, y)[0, 1]

    # print regression statistics
    print("\nLinear Regression Results (Individual Data Points):")
    print(f"Slope: {slope:.4f}")
    print(f"Intercept: {intercept:.4f}")
    print(f"R-squared: {r_value ** 2:.4f}")
    print(f"Correlation Coefficient (r): {correlation_individual:.4f}")
    print(f"p-value: {p_value:.6f}")
    print(f"Standard Error: {std_err:.4f}")

    # plot regression line with scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.3, label="Individual Sentiment Scores")

    # generate regression line
    regression_line = slope * np.array([1, 2, 3, 4, 5]) + intercept
    plt.plot([1, 2, 3, 4, 5], regression_line, 'r', linewidth=2,
             label=f"Regression Line (y = {slope:.3f}x + {intercept:.3f})")

    plt.xlabel("Star Rating")
    plt.ylabel("Sentiment Score")
    plt.title(f"Sentiment Score vs. Rating (Individual Points, R² = {r_value ** 2:.3f})")
    plt.grid(True)
    plt.legend()
    plt.savefig("sentiment_regression_individual.png")
    plt.show()

    # linear regression analysis on average sentiment values
    # prepare data for regression on averages
    x_avg = sentiment_by_rating["rating"]
    y_avg = sentiment_by_rating["sentiment"]

    # calculate the linear regression on averages
    slope_avg, intercept_avg, r_value_avg, p_value_avg, std_err_avg = stats.linregress(x_avg, y_avg)

    # calculate correlation coefficient for averages
    correlation_avg = np.corrcoef(x_avg, y_avg)[0, 1]

    # crint regression statistics for averages
    print("\nLinear Regression Results (Average Sentiment by Rating):")
    print(f"Slope: {slope_avg:.4f}")
    print(f"Intercept: {intercept_avg:.4f}")
    print(f"R-squared: {r_value_avg ** 2:.4f}")
    print(f"Correlation Coefficient (r): {correlation_avg:.4f}")
    print(f"p-value: {p_value_avg:.6f}")
    print(f"Standard Error: {std_err_avg:.4f}")

    # plot regression line with scatter plot of averages
    plt.figure(figsize=(10, 6))
    plt.scatter(x_avg, y_avg, s=100, color='blue', label="Average Sentiment by Rating")

    # generate regression line for averages
    avg_regression_line = slope_avg * np.array([1, 2, 3, 4, 5]) + intercept_avg
    plt.plot([1, 2, 3, 4, 5], avg_regression_line, 'red', linewidth=2,
             label=f"Regression Line (y = {slope_avg:.3f}x + {intercept_avg:.3f})")

    # add data labels for each point
    for i, (rating, sentiment) in enumerate(zip(x_avg, y_avg)):
        plt.annotate(f"{sentiment:.3f}",
                     (rating, sentiment),
                     textcoords="offset points",
                     xytext=(0, 10),
                     ha='center')

    plt.xlabel("Star Rating")
    plt.ylabel("Average Sentiment Score")
    plt.title(f"Average Sentiment Score vs. Rating (R² = {r_value_avg ** 2:.3f})")
    plt.xticks(x_avg)
    plt.grid(True)
    plt.legend()
    plt.savefig("sentiment_regression_averages.png")
    plt.show()


if __name__ == "__main__":
    main()

