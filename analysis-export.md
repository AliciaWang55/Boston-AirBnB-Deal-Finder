# Mini-project: Boston Airbnb prices

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score

RANDOM_STATE = 2

DATA_URL = "https://data.insideairbnb.com/united-states/ma/boston/2026-06-15/data/listings.csv.gz"
df_raw = pd.read_csv(DATA_URL)
df = df_raw.copy()

df.head(10)

## Data card

**Dataset:** Inside Airbnb, Boston listings, scrape of 15 June 2026
**Source:** Inside Airbnb (insideairbnb.com), licensed CC BY 4.0
**Unit of observation:** one row = one Airbnb listing in Boston
**This file:** 4,378 rows × 90 columns
**Target:** `price`, the nightly price of the listing (US dollars)

**Listing and scrape**

| column | meaning |
|---|---|
| `id` | Airbnb's identifier for the listing |
| `listing_url` | link to the listing page |
| `scrape_id` | identifier for the Inside Airbnb scrape run |
| `last_scraped` | date the listing was scraped |
| `source` | whether the listing was found in this scrape ("city scrape") or carried over from a previous one ("previous scrape") |
| `name` | listing title |
| `description` | listing description written by the host |
| `neighborhood_overview` | host's description of the neighbourhood |
| `picture_url` | link to the listing's main photo |

**Host**

| column | meaning |
|---|---|
| `host_id` | Airbnb's identifier for the host |
| `host_url` | link to the host's page |
| `host_profile_id` | identifier for the host's profile |
| `host_profile_url` | link to the host's profile |
| `host_name` | host's first name or business name |
| `host_since` | date the host joined Airbnb |
| `hosts_time_as_user_years` | how long the host has been an Airbnb user: whole years |
| `hosts_time_as_user_months` | how long the host has been an Airbnb user: months beyond the whole years |
| `hosts_time_as_host_years` | how long the host has been hosting: whole years |
| `hosts_time_as_host_months` | how long the host has been hosting: months beyond the whole years |
| `host_location` | where the host says they live |
| `host_about` | host's self-description |
| `host_response_time` | how quickly the host typically responds to messages |
| `host_response_rate` | share of messages the host responds to |
| `host_acceptance_rate` | share of booking requests the host accepts |
| `host_is_superhost` | whether the host has superhost status (t/f) |
| `host_thumbnail_url` | link to the host's small profile photo |
| `host_picture_url` | link to the host's profile photo |
| `host_neighbourhood` | neighbourhood the host lives in |
| `host_listings_count` | number of listings the host has, according to Airbnb |
| `host_total_listings_count` | total number of listings the host has, according to Airbnb |
| `host_verifications` | ways the host has verified their identity (email, phone, ...) |
| `host_has_profile_pic` | whether the host has a profile photo (t/f) |
| `host_identity_verified` | whether the host's identity is verified (t/f) |

**Location**

| column | meaning |
|---|---|
| `neighbourhood` | neighbourhood as entered by the host |
| `neighbourhood_cleansed` | Boston neighbourhood the listing is in, worked out from its coordinates |
| `neighbourhood_group_cleansed` | larger area grouping of neighbourhoods |
| `latitude` | latitude (approximate: Airbnb shifts locations slightly for privacy) |
| `longitude` | longitude (approximate) |

**Property**

| column | meaning |
|---|---|
| `property_type` | kind of property (entire rental unit, private room in home, ...) |
| `room_type` | Entire home/apt, Private room, Hotel room, or Shared room |
| `accommodates` | maximum number of guests |
| `bathrooms` | number of bathrooms |
| `bathrooms_text` | bathrooms as described on the listing (e.g. "1 shared bath") |
| `bedrooms` | number of bedrooms |
| `beds` | number of beds |
| `amenities` | list of amenities the host says the listing has |

**Price**

| column | meaning |
|---|---|
| `price` | **target:** price per night, in US dollars, from a price quote for a specific stay |
| `price_quote_checkin_date` | check-in date of the quoted stay |
| `price_quote_checkout_date` | check-out date of the quoted stay |
| `price_quote_total_price` | total price for the quoted stay, in US dollars |
| `price_quote_price_per_night` | price per night for the quoted stay, in US dollars |
| `price_quote_raw` | the full price quote as returned by Airbnb (JSON text, including fees and discounts) |

**Stay rules and availability**

| column | meaning |
|---|---|
| `minimum_nights` | minimum stay length, in nights |
| `maximum_nights` | maximum stay length, in nights |
| `minimum_minimum_nights` | smallest minimum-stay setting in the calendar over the next year |
| `maximum_minimum_nights` | largest minimum-stay setting in the calendar over the next year |
| `minimum_maximum_nights` | smallest maximum-stay setting in the calendar over the next year |
| `maximum_maximum_nights` | largest maximum-stay setting in the calendar over the next year |
| `minimum_nights_avg_ntm` | average minimum-stay setting over the next 12 months |
| `maximum_nights_avg_ntm` | average maximum-stay setting over the next 12 months |
| `calendar_updated` | when the host last updated the calendar |
| `has_availability` | whether the listing has any availability (t/f) |
| `availability_30` | nights available to book in the next 30 days |
| `availability_60` | nights available to book in the next 60 days |
| `availability_90` | nights available to book in the next 90 days |
| `availability_365` | nights available to book in the next 365 days |
| `calendar_last_scraped` | date the calendar was scraped |

**Reviews and activity**

| column | meaning |
|---|---|
| `number_of_reviews` | total number of reviews |
| `number_of_reviews_ltm` | reviews in the last 12 months |
| `number_of_reviews_l30d` | reviews in the last 30 days |
| `availability_eoy` | nights available to book between the scrape and the end of the year |
| `number_of_reviews_ly` | reviews in the last year |
| `estimated_occupancy_l365d` | Inside Airbnb's estimate of nights booked in the last 365 days |
| `estimated_revenue_l365d` | Inside Airbnb's estimate of revenue in the last 365 days, in US dollars |
| `first_review` | date of the first review |
| `last_review` | date of the most recent review |
| `review_scores_rating` | overall guest rating (1 to 5) |
| `review_scores_accuracy` | guest rating for accuracy of the listing (1 to 5) |
| `review_scores_cleanliness` | guest rating for cleanliness (1 to 5) |
| `review_scores_checkin` | guest rating for check-in (1 to 5) |
| `review_scores_communication` | guest rating for communication (1 to 5) |
| `review_scores_location` | guest rating for location (1 to 5) |
| `review_scores_value` | guest rating for value (1 to 5) |

**Other**

| column | meaning |
|---|---|
| `license` | short-term rental licence, registration, or exemption the host entered |
| `instant_bookable` | whether guests can book without the host approving first (t/f) |
| `calculated_host_listings_count` | number of listings this host has in this Boston file |
| `calculated_host_listings_count_entire_homes` | of those, how many are entire homes |
| `calculated_host_listings_count_private_rooms` | of those, how many are private rooms |
| `calculated_host_listings_count_shared_rooms` | of those, how many are shared rooms |
| `reviews_per_month` | average number of reviews per month |

**Collection notes:** Inside Airbnb scrapes the public Airbnb website roughly every quarter. Prices come from a quote for a particular set of dates, so they can include discounts for longer stays. Everything about a listing is what the host entered or what Airbnb displayed, not independently verified.

## Research question

How many Airbnbs are priced higher or lower relative to other listings with similar characteristics (accomodates, room_type, property_type, etc) and how does that relate to ratings (excluding the rating metric about value)? Trying to determine how many "good deals" (low price but high rating, so not exactly in terms of price) there are and if there are common listing characteristics between those good deals. Last part will be done using logistic regression to predict good deals.

## Data cleaning

df.shape

df.describe()

missing = df.isnull().sum()
missing[missing>0]

There are many columns where there is no data at all, indicated by 4378 missing values. First, drop these columns because there is no good way to deal with features that are completely missing. In these columns, there are no data to predict price with. Also remove id columns because they provide no meaningful information.

df.drop(columns = ["id", "scrape_id", "neighborhood_overview", "neighbourhood", "host_since", "host_response_time", "host_response_rate", "host_acceptance_rate", "host_thumbnail_url", "host_total_listings_count", "instant_bookable", "calendar_updated", "neighbourhood_group_cleansed", "host_verifications", "host_neighbourhood"], inplace = True)
df

missing = df.isnull().sum()
missing[missing>0]

#convert price to numeric
df["price"] = pd.to_numeric(df["price"].str.replace("[$,]", "", regex=True))
#remove any duplicates
df = df.drop_duplicates(subset="listing_url")
df.shape

## EDA

#Look at distribution of bathrooms and beds and bedrooms
sns.histplot(df["bathrooms"])
plt.show()
print("mean:", df["bathrooms"].mean())
print("median:", df["bathrooms"].median())
print("skew:", df["bathrooms"].skew())

sns.histplot(df["beds"])
plt.show()
print("mean:", df["beds"].mean())
print("median:", df["beds"].median())
print("skew:", df["beds"].skew())

sns.histplot(df["bedrooms"])
plt.show()
print("mean:", df["bedrooms"].mean())
print("median:", df["bedrooms"].median())
print("skew:", df["bedrooms"].skew())

#Fill in bathroom, beds, bathrooms with median, because all distributions heavily right-skewed so mean is not a good choice
df["bathrooms"] = df["bathrooms"].fillna(df["bathrooms"].median())
df["beds"] = df["beds"].fillna(df["beds"].median())
df["bedrooms"] = df["bedrooms"].fillna(df["bedrooms"].median())

#Look at price distribution
sns.histplot(df["price"])
plt.show()
print("mean:", df["price"].mean())
print("median:", df["price"].median())
print("skew:", df["price"].skew())

#df["price"] = df["price"].fillna(df["price"].median())
df = df.dropna(subset=["price"])

df["log_price"] = np.log(df["price"]) #reduce influence of extreme prices

#filter so that only consider rows with more than 1 reviews
df = df[df["number_of_reviews"] > 1]
df.shape

#log price boxplot
sns.boxplot(data = df, x="log_price")

sns.histplot(data = df, x="review_scores_rating")

sns.boxplot(data = df, x="review_scores_rating")

sns.scatterplot(data=df, x="review_scores_rating", y="log_price", alpha=0.4)
plt.show()

Describing this scatterplot:
- Heavy left-skew in review ratings
- No linear correlation
- However, lower-rated listings are almost nonexistent at higher price points
- Wide price variance at high ratings
- Variance in price (log_price) increases as review ratings increase

print(df["review_scores_rating"].describe())
print(df["review_scores_rating"].quantile([0.25, 0.5, 0.75, 0.9]))

sns.scatterplot(data=df, x="accommodates", y="log_price")

sns.scatterplot(data=df, x="minimum_nights", y="log_price")

sns.boxplot(data=df, x="room_type", y="log_price")

df["room_type"].value_counts()

# make new columns for each room type (except Entire home/apt)
df["is_private_room"] = (df["room_type"] == "Private room").astype(int)
df["is_shared_room"] = (df["room_type"] == "Shared room").astype(int)
df["is_hotel_room"] = (df["room_type"] == "Hotel room").astype(int)

# 0/1 columns for the 20 most common neighborhoods (all others are the baseline)
nbhd_cols = []
for n in df["neighbourhood_cleansed"].value_counts().head(20).index:
    df["nbhd_" + n] = (df["neighbourhood_cleansed"] == n).astype(int)
    nbhd_cols.append("nbhd_" + n)

cols = ["log_price", "review_scores_rating", "accommodates", "bathrooms",
             "bedrooms", "beds", "minimum_nights", "latitude", "longitude",
             "is_private_room", "is_shared_room", "is_hotel_room"]
sns.heatmap(df[cols].corr(), annot=True)

df["long_stay"] = (df["minimum_nights"] >= 28).astype(int)
print(df["long_stay"].mean())

prop_cols = []
for p in df["property_type"].value_counts().head(8).index:
    df["prop_" + p] = (df["property_type"] == p).astype(int)
    prop_cols.append("prop_" + p)

amenity_search = {"has_ac": "air conditioning", "has_free_parking": "free parking",
                  "has_paid_parking": "paid parking", "has_dishwasher": "dishwasher",
                  "has_elevator": "elevator", "has_gym": "gym",
                  "has_patio": "patio or balcony", "has_self_checkin": "self check-in"}
amen_cols = []
for col in amenity_search:
    df[col] = df["amenities"].str.lower().str.contains(amenity_search[col], regex=False).astype(int)
    amen_cols.append(col)

print(df[amen_cols].mean())   # share of listings with each

## Modeling

X = df[["accommodates", "bathrooms", "bedrooms", "beds", "minimum_nights", "latitude", "longitude",
        "long_stay", "is_private_room", "is_shared_room", "is_hotel_room"] + nbhd_cols + amen_cols + prop_cols]
y = df["log_price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

model = LinearRegression().fit(X_train, y_train)
print(model.score(X_test, y_test))

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("Training R²:", round(train_r2, 4))
print("Testing R²:", round(test_r2, 4))

The model explains 82.81% of the variation in log prices in the training data.

The model explains 81.39% of the variation in log prices in the unseen test data.

# distribution of differences between actual and predicted log prices across all listings
df["pred_log_price"] = model.predict(X)
df["residual"] = df["log_price"] - df["pred_log_price"]
sns.histplot(df["residual"])

for n in df["neighbourhood_cleansed"].value_counts().head(10).index:
    print(n, round(df[df["neighbourhood_cleansed"] == n]["residual"].median(), 3))

resid_cutoff = 0.2    # log scale: about 18% below or 22% above the expected price

df["price_relative"] = "fair"   # default
df.loc[df["residual"] < -resid_cutoff, "price_relative"] = "underpriced"
df.loc[df["residual"] > resid_cutoff, "price_relative"] = "overpriced"

print(df["price_relative"].value_counts())
print(df["price_relative"].value_counts(normalize=True))   # percentages

Answer to research question: How many Airbnbs are priced higher or lower relative to other listings with similar characteristics?

806 (28%) underpriced, 1325 (46%) fair, 736 (26%)overpriced

print(df[["log_price", "review_scores_rating"]].corr())
print(df[["residual", "review_scores_rating"]].corr())

Very weak relationship between log_price and review_scores_rating, also shown in scatterplot above

Listings priced below their predicted prices (underpriced/negative residual) do not automatically have higher ratings

rating_cutoff = 4.82   # to determine relatively good ratings vs relatively "bad" ratings, around 50% percentile based off the numbers from rating describe

df["good_deal"] = ((df["price_relative"] == "underpriced") &
                   (df["review_scores_rating"] >= rating_cutoff)).astype(int)
print(df["good_deal"].value_counts())
print("share of good deals:", df["good_deal"].mean())

374 listings (13% of total) where the the price is more than 18% below expected price and the rating is around/above the 50th percentile.

# no rating, residual, or price
# latitude and longitude are dropped because the neighborhood columns replace them
other_cols = [
    "accommodates", "bathrooms", "bedrooms", "beds",
    "minimum_nights", "long_stay",
    "is_private_room", "is_shared_room", "is_hotel_room",
    "number_of_reviews", "calculated_host_listings_count"
]

df["has_license"] = df["license"].notna().astype(int)

extra_cols = [
    "has_license", "availability_365", "reviews_per_month"
]

X = df[other_cols + extra_cols + nbhd_cols + amen_cols + prop_cols]
y = df["good_deal"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y # keep roughly same proportion of good deals in each set, important bc only 13% are good deals
)

print("Good deals in train:", y_train.sum(),
      "| in test:", y_test.sum())

# Logistic Regression
log_model = LogisticRegression(
    max_iter=5000,
    class_weight="balanced" #increase relative importance of less common class (good deals)
)
log_model.fit(X_train, y_train)

# Decision Tree
tree_model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_leaf=20,
    class_weight="balanced",
    random_state=RANDOM_STATE
)
tree_model.fit(X_train, y_train)

# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=300, #300 trees
    max_depth=8,
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

from sklearn.model_selection import StratifiedKFold, cross_val_score

kf = StratifiedKFold( # for classification task
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

log_cv = LogisticRegression(
    max_iter=5000,
    class_weight="balanced"
)

log_precision = cross_val_score(
    log_cv, X, y, cv=kf, scoring="precision"
)
log_recall = cross_val_score(
    log_cv, X, y, cv=kf, scoring="recall"
)
log_f1 = cross_val_score(
    log_cv, X, y, cv=kf, scoring="f1"
)

print("Logistic Regression")
print("Precision:", log_precision.mean())
print("Recall:", log_recall.mean())
print("F1:", log_f1.mean())

tree_cv = DecisionTreeClassifier(
    max_depth=5,
    min_samples_leaf=20,
    class_weight="balanced",
    random_state=RANDOM_STATE
)

tree_precision = cross_val_score(
    tree_cv, X, y, cv=kf, scoring="precision"
)
tree_recall = cross_val_score(
    tree_cv, X, y, cv=kf, scoring="recall"
)
tree_f1 = cross_val_score(
    tree_cv, X, y, cv=kf, scoring="f1"
)

print("Decision Tree")
print("Precision:", tree_precision.mean())
print("Recall:", tree_recall.mean())
print("F1:", tree_f1.mean())

rf_cv = RandomForestClassifier()

rf_precision = cross_val_score(
    rf_cv, X, y, cv=kf, scoring="precision"
)
rf_recall = cross_val_score(
    rf_cv, X, y, cv=kf, scoring="recall"
)
rf_f1 = cross_val_score(
    rf_cv, X, y, cv=kf, scoring="f1"
)

print("Random Forest")
print("Precision:", rf_precision.mean())
print("Recall:", rf_recall.mean())
print("F1:", rf_f1.mean())

models = {
    "Logistic Regression": log_model,
    "Decision Tree": tree_model,
    "Random Forest": rf_model
}

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

for name, model in models.items():
    print(f"\n{name}")

    probabilities = model.predict_proba(X_test)[:, 1]

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

        print(
            f"Threshold: {threshold:.1f} | "
            f"Accuracy: {accuracy_score(y_test, predictions):.3f} | "
            f"Precision: {precision_score(y_test, predictions, zero_division=0):.3f} | "
            f"Recall: {recall_score(y_test, predictions, zero_division=0):.3f} | "
            f"F1: {f1_score(y_test, predictions, zero_division=0):.3f}"
        )

Prioritizing precision because I think it'd be more helpful if the good deals that the model predicts are real good deals rather than trying to classify as many good deals as there are.

Random Forest Threshold 0.6

# predict probabilities on the test set
# final test predictions
rf_probs = rf_model.predict_proba(X_test)[:, 1]

threshold = 0.6
rf_preds = (rf_probs >= threshold).astype(int)

## Validation

print("share of good deals in test set:", y_test.mean())
print("accuracy:", accuracy_score(y_test, rf_preds))
print("precision:", precision_score(y_test, rf_preds))
print("recall:", recall_score(y_test, rf_preds))
print(confusion_matrix(y_test, rf_preds))

deals = df[df["good_deal"] == 1]
others = df[df["good_deal"] == 0]

data = {"good deals": deals[X.columns].mean(),
        "all others": others[X.columns].mean()}

compare = pd.DataFrame(data)
compare["Difference"] = compare["good deals"] - compare["all others"]
print(compare)

log_coefficients = pd.DataFrame({"Feature": X_train.columns,
                     "Coefficient": log_model.coef_[0],
                     "Odds ratio": np.exp(log_model.coef_[0])}).sort_values(by="Odds ratio", ascending=False)
print(log_coefficients.round(3))

Logistic regression estimates how each listing characteristic changes the odds that a listing is a good deal, holding the other characteristics fixed. An odds ratio above 1 means good deals are more likely with that feature, and below 1 means less likely.

## Conclusion

I built a linear regression model to predict prices based on characteristics such as neighborhood, room type, size, and amenities, achieving an (R^2) of 0.814 on the test set. I compared each listing’s actual price with its predicted price, classifying 28.1% as underpriced, 46.2% as fairly priced, and 25.7% as overpriced using a residual threshold of ±0.2 log-price units. I defined a “good deal” as an underpriced listing with a guest rating of at least 4.82, identifying 374 good deals, or approximately 13.0% of the dataset. Using logistic regression, a decision tree, and a random forest, I explored whether listing characteristics could predict good deals; my analysis found that neighborhood was relevant, with the logistic regression model showing higher estimated odds of a listing being a good deal in West Roxbury (odds ratio 2.64) and Mattapan (odds ratio 2.38), relative to the model’s reference neighborhood, holding other included features constant.

###Heatmap

# columns needed for mapping
map_df = df[
    [
        "latitude",
        "longitude",
        "price",
        "price_relative",
        "good_deal",
        "review_scores_rating",
        "room_type",
        "property_type",
        "neighbourhood_cleansed"
    ]
].copy()

map_df = map_df.dropna(subset=["latitude", "longitude"])

map_df = map_df.rename(columns={
    "latitude": "point_latitude",
    "longitude": "point_longitude"
})

map_df.to_csv("boston_airbnb_good_deals.csv", index=False)

print("Listings exported:", len(map_df))
print(map_df.head())

from google.colab import files
files.download("boston_airbnb_good_deals.csv")



