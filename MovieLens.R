################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))), col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId], title = as.character(title), genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")

test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]
 
# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)

edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# END OF LOADING SCRIPT

# Examine what data we have
knitr::kable(
  edx[1:5,4:6]
)

# START CLEANING DATA
library(stringr)

#First we must add a year column to the edx and validation set
edx <- edx %>% 
  mutate(year = as.numeric(str_match(title, "\\((\\d{4})\\)$")[,2]))

validation <- validation %>% 
  mutate(year = as.numeric(str_match(title, "\\((\\d{4})\\)$")[,2]))

#Examine cleaned data
knitr::kable(
  edx[1:5,4:7]
)

#Now we need to split the working set into a test set and a training set
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, 
                                  list = FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]

test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# To assess the quality of our model we need to 
# calculate the Root Mean Squared Error

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Simplest possible model Y_u,i = mu + epsilon_u,i
mu <- mean(train_set$rating)
#We can see what RMSE this gives us: 1.059909
RMSE(test_set$rating, mu)

# Looking at the movie effect adjusts the model: Y_u,i = mu + b_i + epsilon_u,i
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

# Review the bias data
knitr::kable(
  movie_avgs[1:5,]
)

# We can visualize the variation in the average ratings of movies
movie_avgs %>% ggplot(aes(b_i)) + 
  geom_histogram(bins=30, color="white", fill="cyan3")

# Review the correlation of the mean rating and the bias - unsuprisingly it is very high
movie_avgs %>% filter(movieId <= 20) %>% 
  left_join(train_set, by='movieId') %>% 
  group_by(b_i) %>% 
  summarize(mean_rating=mean(rating)) %>%
  ggplot(aes(b_i, mean_rating)) +
  geom_point(color="cyan3", fill="cyan3")

# Now we can calculate new predictions, using the movie effect
predicted_ratings <- mu + test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)

#We see we get a better RMSE:0.9437429
RMSE(test_set$rating, predicted_ratings)

# There may also be a user effect: Y_u,i = mu + b_i + b_u + epsilon_u,i
# We can visualize how ratings vary by user for active users
train_set %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating - mu)) %>% 
  filter(n()>=100) %>%
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "white", fill="cyan3")

# See the correlation for the first 20 users
user_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

user_avgs %>% filter(userId <= 20) %>% 
  left_join(train_set, by='userId') %>%
  left_join(movie_avgs, by='movieId') %>%
  group_by(b_u) %>% 
  summarize(mean_rating=mean(rating)) %>%
  ggplot(aes(b_u, mean_rating)) +
  geom_point(color="cyan3", fill="cyan3")

# So now we can retreive users' average ratings 
# and use them to create new predictions

predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

# Once again, we get an improved RMSE: 0.865932
RMSE(test_set$rating, predicted_ratings)

# There may also be a genre effect: Y_u,i,g = mu + b_i + b_u + b_g + epsilon_u,i,g
#  We can visualize how ratings vary by genre

#First lets see how many genre combinations there are: 797
train_set %>% group_by(genres) %>% summarize(n()) %>% nrow()

#Now lets see how rating varies by genre combination
train_set %>% 
  group_by(genres) %>% 
  summarize(b_g = mean(rating - mu)) %>%
  ggplot(aes(b_g)) + 
  geom_histogram(bins = 30, color = "white", fill="cyan3")

# Review the correlation
genre_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu - b_i - b_u))

cor_check <- genre_avgs %>%  
  left_join(train_set, by='genres') %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  filter(movieId <= 100) %>%
  filter(userId <= 100) 

cor_check %>%
  ggplot(aes(b_g, b_i)) +
  geom_point(color="cyan3", fill="cyan3")

cor(cor_check$b_g, cor_check$b_i)

# So now we can retreive genre combination's average ratings 
# and use them to create new predictions

predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  mutate(pred = mu + b_i + b_u + b_g) %>%
  pull(pred)

# Once again, we get an improved RMSE: 0.8655941
RMSE(test_set$rating, predicted_ratings)

# There may also be a year of release effect: Y_u,i,g,y = mu + b_i + b_u + b_g + b_y + epsilon_u,i,g,y
#  We can visualize how ratings vary by year of release

# First lets see how many release years there are: 94
train_set %>% group_by(genres) %>% summarize(n()) %>% nrow()

#Now lets see how rating varies by year of release

train_set %>% 
  group_by(year) %>% 
  summarize(b_y = mean(rating - mu)) %>%
  ggplot(aes(year, b_y)) + 
  geom_bar(stat="identity", color = "white", fill="cyan3")

# So now we can retreive each release year's average ratings 
# and use them to create new predictions

year_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  group_by(year) %>%
  summarize(b_y = mean(rating - mu - b_i - b_u - b_g))

predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  left_join(year_avgs, by='year') %>%
  mutate(pred = mu + b_i + b_u + b_g + b_y) %>%
  pull(pred)

# Once again, we get an improved RMSE: 0.8654189
RMSE(test_set$rating, predicted_ratings)

# Next we can look at how the rating varies by number of people rating a film

train_set %>%
  group_by(movieId) %>%
  summarize(n = n(), rating=mean(rating)) %>%
  group_by(n) %>%
  summarize(b_n = mean(rating - mu)) %>%
  ggplot(aes(n, b_n)) + 
  geom_bar(stat="identity", color = "cyan3", fill="cyan3")

# It's more obvious with more smoothing (wider bins)
train_set %>%
  group_by(movieId) %>%
  summarize(n = round(n(), -2), rating=mean(rating)) %>%
  group_by(n) %>%
  summarize(b_n = mean(rating - mu)) %>%
  ggplot(aes(n, b_n)) + 
  geom_bar(stat="identity", color = "cyan3", fill="cyan3")
  
# So we can see there could be a popularity effect: 
# Y_u,i,g,y,n = mu + b_i + b_u + b_g + b_y + b_n + epsilon_u,i,g,y,n

# Let's first calculate movie popularity
# we'll round to the nearest hundred to avoid having too many categories
movie_pop <- train_set %>%
  group_by(movieId) %>%
  summarize(num_ratings=round(n(), -2)) 

pop_avgs <- train_set %>%
  left_join(movie_pop, by='movieId') %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  left_join(year_avgs, by='year') %>%
  group_by(num_ratings) %>%
  summarize(b_n = mean(rating - mu - b_i - b_u - b_g - b_y))

predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  left_join(year_avgs, by='year') %>%
  left_join(movie_pop, by='movieId') %>%
  left_join(pop_avgs, by='num_ratings') %>%
  mutate(pred = mu + b_i + b_u + b_g + b_y + b_n) %>%
  pull(pred)

# Once again, we get an improved RMSE: 0.8652957
RMSE(test_set$rating, predicted_ratings)

# Next we can look at the effect on rating of the number of days 
# since a movie was first rated
# First let's calculate a table of the number of days since each movie was first rated
# We'll round to the nearest 10 to get a sensible number of groups
movie_days <- train_set %>% 
  mutate(days_since=as.numeric(
                      as.Date("2020-04-30") - 
                      as.Date(as.POSIXct(timestamp, origin="1970-01-01")))) %>%
  group_by(movieId) %>%
  summarize(movie_days_since=round(max(days_since),-1))

movie_day_avgs <- train_set %>%
  left_join(movie_days, by='movieId') %>%
  left_join(movie_pop, by='movieId') %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  left_join(year_avgs, by='year') %>%
  left_join(pop_avgs, by='num_ratings') %>%
  group_by(movie_days_since) %>%
  summarize(b_md = mean(rating - mu - b_i - b_u - b_g - b_y - b_n))

# Let's see how the bias varies with the number of days since a movie was first rated
movie_day_avgs %>%
  ggplot(aes(movie_days_since, b_md)) +
  geom_point(color="cyan3", fill="cyan3") +
  geom_smooth()

# Now we can add to the model and make new predictions

predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  left_join(year_avgs, by='year') %>%
  left_join(movie_pop, by='movieId') %>%
  left_join(pop_avgs, by='num_ratings') %>%
  left_join(movie_days, by='movieId') %>%
  left_join(movie_day_avgs, by='movie_days_since') %>%
  mutate(pred = mu + b_i + b_u + b_g + b_y + b_n + b_md) %>%
  pull(pred)

# Once again, we get an improved RMSE: 0.8652373
RMSE(test_set$rating, predicted_ratings)

# OK, now we can look at the effect on rating of the number of days
# since a user first rated any movie
# First let's calculate a table of the number of days since each user first rated a movie
# We'll round to the nearest 10 to get a sensible number of groups
user_days <- train_set %>% 
  mutate(days_since=as.numeric(
    as.Date("2020-04-30") - 
      as.Date(as.POSIXct(timestamp, origin="1970-01-01")))) %>%
  group_by(userId) %>%
  summarize(user_days_since=round(max(days_since),-1))

# Let's look at the distribution
user_days %>% ggplot(aes(user_days_since)) + 
  geom_histogram(bins=30, color="white", fill="cyan3")


# Now we can calculate the bias by user
user_day_avgs <- train_set %>%
  left_join(user_days, by='userId') %>%
  left_join(movie_days, by='movieId') %>%
  left_join(movie_pop, by='movieId') %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  left_join(year_avgs, by='year') %>%
  left_join(pop_avgs, by='num_ratings')%>%
  left_join(movie_day_avgs, by='movie_days_since') %>%
  group_by(user_days_since) %>%
  summarize(b_ud = mean(rating - mu - b_i - b_u - b_g - b_y - b_n - b_md))

# The strength of the effect is really clear from the plot
user_day_avgs %>%
  ggplot(aes(user_days_since, b_ud)) +
  geom_point(color="cyan3", fill="cyan3") + 
  geom_smooth()

# So we can add to the model to improve our accuracy
predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  left_join(year_avgs, by='year') %>%
  left_join(movie_pop, by='movieId') %>%
  left_join(pop_avgs, by='num_ratings') %>%
  left_join(movie_days, by='movieId') %>%
  left_join(movie_day_avgs, by='movie_days_since') %>%
  left_join(user_days, by='userId') %>%
  left_join(user_day_avgs, by='user_days_since') %>%
  mutate(pred = mu + b_i + b_u + b_g + b_y + b_n + b_md + b_ud) %>%
  pull(pred)

# Once again, we get an improved RMSE: 0.8651212
RMSE(test_set$rating, predicted_ratings)

# Regularization is an important part of the model process, this plot shows why
train_set %>% 
  group_by(movieId) %>%
  summarize(mean_rating=mean(rating)) %>% 
  left_join(movie_pop, by='movieId') %>%
  filter(movieId <= 1000) %>%
  ggplot(aes(num_ratings, mean_rating)) + 
  geom_point(color="cyan3", fill="cyan3", alpha=0.1) +
  geom_smooth()

# Now we can try an approach using regularized values for the predictors (we're calculating the means manually, adding lambda to the divisor)

# We'll try 40 different values of lambda to find the optimum
lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){

movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+l))

user_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_i)/(n()+l))
  
genre_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - mu - b_i - b_u)/(n()+l))

year_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  group_by(year) %>%
  summarize(b_y = sum(rating - mu - b_i - b_u - b_g)/(n()+l))

movie_pop <- train_set %>%
  group_by(movieId) %>%
  summarize(num_ratings=round(n(), -2)) 

pop_avgs <- train_set %>%
  left_join(movie_pop, by='movieId') %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  left_join(year_avgs, by='year') %>%
  group_by(num_ratings) %>%
  summarize(b_n = sum(rating - mu - b_i - b_u - b_g - b_y)/(n()+l))

movie_days <- train_set %>% 
  mutate(days_since=as.numeric(
    as.Date("2020-04-30") - 
      as.Date(as.POSIXct(timestamp, origin="1970-01-01")))) %>%
  group_by(movieId) %>%
  summarize(movie_days_since=round(max(days_since),-1))

movie_day_avgs <- train_set %>%
  left_join(movie_days, by='movieId') %>%
  left_join(movie_pop, by='movieId') %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  left_join(year_avgs, by='year') %>%
  left_join(pop_avgs, by='num_ratings') %>%
  group_by(movie_days_since) %>%
  summarize(b_md = sum(rating - mu - b_i - b_u - b_g - b_y - b_n)/(n()+l))

user_days <- train_set %>% 
  mutate(days_since=as.numeric(
    as.Date("2020-04-30") - 
      as.Date(as.POSIXct(timestamp, origin="1970-01-01")))) %>%
  group_by(userId) %>%
  summarize(user_days_since=round(max(days_since),-1))

user_day_avgs <- train_set %>%
  left_join(user_days, by='userId') %>%
  left_join(movie_days, by='movieId') %>%
  left_join(movie_pop, by='movieId') %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  left_join(year_avgs, by='year') %>%
  left_join(pop_avgs, by='num_ratings')%>%
  left_join(movie_day_avgs, by='movie_days_since') %>%
  group_by(user_days_since) %>%
  summarize(b_ud = sum(rating - mu - b_i - b_u - b_g - b_y - b_n - b_md)/(n()+l))

predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  left_join(year_avgs, by='year') %>%
  left_join(movie_pop, by='movieId') %>%
  left_join(pop_avgs, by='num_ratings') %>%
  left_join(movie_days, by='movieId') %>%
  left_join(movie_day_avgs, by='movie_days_since') %>%
  left_join(user_days, by='userId') %>%
  left_join(user_day_avgs, by='user_days_since') %>%
  mutate(pred = mu + b_i + b_u + b_g + b_y + b_n + b_md + b_ud) %>%
  pull(pred)

return(RMSE(predicted_ratings, test_set$rating))
})
# Plotting the RMSE for each lambda shows us where the minimum RMSE is to be found
qplot(lambdas, rmses)  

# So which lambda minimizes the RMSE? 4.75
lambdas[which.min(rmses)]

#The RMSE returned with a lambda of 4.75 on the test set is 0.8645087
min(rmses)

# So we've established that the optimum lambda is 4.75. 
# Now we need to test using that lambda on the verification set.

# Now we can start with the predictions

l <- 4.75

movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+l))

user_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_i)/(n()+l))

genre_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - mu - b_i - b_u)/(n()+l))

year_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  group_by(year) %>%
  summarize(b_y = sum(rating - mu - b_i - b_u - b_g)/(n()+l))

movie_pop <- edx %>%
  group_by(movieId) %>%
  summarize(num_ratings=round(n(), -2)) 

pop_avgs <- edx %>%
  left_join(movie_pop, by='movieId') %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  left_join(year_avgs, by='year') %>%
  group_by(num_ratings) %>%
  summarize(b_n = sum(rating - mu - b_i - b_u - b_g - b_y)/(n()+l))

movie_days <- edx %>% 
  mutate(days_since=as.numeric(
    as.Date("2020-04-30") - 
      as.Date(as.POSIXct(timestamp, origin="1970-01-01")))) %>%
  group_by(movieId) %>%
  summarize(movie_days_since=round(max(days_since),-1))

movie_day_avgs <- edx %>%
  left_join(movie_days, by='movieId') %>%
  left_join(movie_pop, by='movieId') %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  left_join(year_avgs, by='year') %>%
  left_join(pop_avgs, by='num_ratings') %>%
  group_by(movie_days_since) %>%
  summarize(b_md = sum(rating - mu - b_i - b_u - b_g - b_y - b_n)/(n()+l))

user_days <- edx %>% 
  mutate(days_since=as.numeric(
    as.Date("2020-04-30") - 
      as.Date(as.POSIXct(timestamp, origin="1970-01-01")))) %>%
  group_by(userId) %>%
  summarize(user_days_since=round(max(days_since),-1))

user_day_avgs <- edx %>%
  left_join(user_days, by='userId') %>%
  left_join(movie_days, by='movieId') %>%
  left_join(movie_pop, by='movieId') %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  left_join(year_avgs, by='year') %>%
  left_join(pop_avgs, by='num_ratings')%>%
  left_join(movie_day_avgs, by='movie_days_since') %>%
  group_by(user_days_since) %>%
  summarize(b_ud = sum(rating - mu - b_i - b_u - b_g - b_y - b_n - b_md)/(n()+l))

# Only here finally, can we join with the validation set to calculate the final predictions and check the RMSE
predicted_ratings <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  left_join(year_avgs, by='year') %>%
  left_join(movie_pop, by='movieId') %>%
  left_join(pop_avgs, by='num_ratings') %>%
  left_join(movie_days, by='movieId') %>%
  left_join(movie_day_avgs, by='movie_days_since') %>%
  left_join(user_days, by='userId') %>%
  left_join(user_day_avgs, by='user_days_since') %>%
  mutate(pred = mu + b_i + b_u + b_g + b_y + b_n + b_md + b_ud) %>%
  pull(pred)

# We now find we have an RMSE on the validation set of 0.8640077  
RMSE(predicted_ratings, validation$rating)



