# Movie Recommendation System using MovieLens Dataset

# Libraries used
if(!require(tidyverse)) install.packages("tidyverse", 
    repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", 
    repos = "http://cran.us.r-project.org")
library(tidyverse)
library(caret)
library(dplyr)
library(ggplot2)
library(wordcloud)
library(knitr)
library(MLmetrics)
library(tinytex)

# Data Preparation
# Load Data
# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip
options(timeout = 120)
dl <- "ml-10M100K.zip"
if(!file.exists(dl)) 
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)
ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)
movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)
# The next step will need comparatively more time for execution
ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE), stringsAsFactors = FALSE)                                                                                                                                                                                     
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>% mutate(userId = as.integer(userId), movieId = as.integer(movieId), rating = as.numeric(rating), timestamp = as.integer(timestamp))
movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE), stringsAsFactors = FALSE)                                                                                              
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>% mutate(movieId = as.integer(movieId))
movielens <- left_join(ratings, movies, by = "movieId")

# Confirming availability of combined dataset as desired
str(movielens)

# Training and testing of the Algorithm : Creating final_holdout_test set

# Final_holdout_test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index, ]
temp <- movielens[test_index, ]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% semi_join(edx, by = "movieId") %>% semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

# removing variables from workspace
rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Dividing edx further into two, one for building the algorithm and another for testing the algorithm
# Creating train and test sets
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_set <- edx[-test_index,]
temp <- edx[test_index,]
# Matching userId and movieId in both train and test sets
test_set <- temp %>% semi_join(train_set, by = "movieId") %>% semi_join(train_set, by = "userId")
# Adding back rows into train set
removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed) 
rm(test_index, temp, removed)

# Analysing the data structure. 
str(edx)

# Understanding the data on summary basis
edx %>% select(-genres) %>% summary()

# finding the unique numbers of users and movie
edx %>% summarize(n_users = n_distinct(userId), n_movies = n_distinct(movieId))

# Drawing a Bar plot of top 20 movies based on popularity for analysis of popularity    
edx %>% group_by(title) %>% summarize(count = n()) %>% arrange(-count) %>% top_n(20, count) %>% ggplot(aes(count, reorder(title, count))) + geom_bar(color = "black", fill = "deepskyblue2", stat = "identity") + xlab("Count") + ylab(NULL) + theme_bw()

# Drawing a Relative Frequency Bar plot of the most given ratings to visulise the pattern
edx %>% ggplot(aes(rating, y = ..prop..)) + geom_bar(color = "black", fill = "deepskyblue2") + labs(x = "Ratings", y = "Relative Frequency") + scale_x_continuous(breaks = c(0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5)) + theme_bw()

# Finding the count of movie ratings for every genre
# The next step will need comparatively much more time for execution
top_genr <- edx %>% separate_rows(genres, sep = "\\|") %>% group_by(genres) %>% summarize(count = n()) %>% arrange(desc(count))
top_genr

# Using word cloud to visualise the movie ratings with most genres.
pal <- brewer.pal(8, "Dark2")
top_genr %>% with(wordcloud(genres, count, max.words = 50, random.order = FALSE, colors = pal))

# Models / Algorithms :

# Model 1 Simply the mean

# Prediction using mean
mu_hat <- mean(train_set$rating)
mu_hat

# Finding the RMSE 
model1_rmse <- RMSE(test_set$rating, mu_hat)
results = tibble(Method = "Model 1: Simply the mean", RMSE = model1_rmse)
results %>% knitr::kable()

# Model 2 Mean + movie bias

bi <- train_set %>% group_by(movieId) %>% summarize(b_i = mean(rating - mu_hat))

# Drawing a histogram to understand the movie bias through the distribution
bi %>% ggplot(aes(b_i)) + geom_histogram(color = 'black', fill = 'deepskyblue2', bins = 10) + xlab('Movie Bias') + ylab('Count') + theme_bw()

# Prediction and Finding the RMSE
predicted_ratings <- mu_hat + test_set %>% left_join(bi, by = "movieId") %>% pull(b_i)
m_bias_rmse <- RMSE(predicted_ratings, test_set$rating)
results <- bind_rows(results, tibble(Method = "Model 2: Mean + movie bias", RMSE = m_bias_rmse))
results %>% knitr::kable()

# Model 3 Mean + movie bias + user effect

bu <- train_set %>% left_join(bi, by = 'movieId') %>% group_by(userId) %>% summarize(b_u = mean(rating - mu_hat - b_i))

# Prediction and Finding the RMSE
predicted_ratings <- test_set %>% left_join(bi, by = 'movieId') %>% left_join(bu, by = 'userId') %>% mutate(pred = mu_hat + b_i + b_u) %>% pull(pred)
u_bias_rmse <- RMSE(predicted_ratings, test_set$rating)
results <- bind_rows(results, tibble(Method = 'Model 3: Mean + movie_bias + user effect', RMSE = u_bias_rmse))
results %>% knitr::kable()

# Model 4 Regularised movie bias and user effect
# Using regularization to penalize large estimates formed by small sample sizes
lambdas <- seq(0, 10, 0.25)
# The next step will need comparatively much more time for execution
rmses <- sapply(lambdas, function(x) {
    	mu_hat <- mean(train_set$rating)
    	b_i <- train_set %>% 
        		group_by(movieId) %>% 
        		summarize(b_i = sum(rating - mu_hat) / (n() + x))
	    b_u <- train_set %>%
		        left_join(b_i, by = 'movieId') %>%
		        group_by(userId) %>%
		        summarize(b_u = sum(rating - b_i - mu_hat) / (n() + x))
    	predicted_ratings <- test_set %>%
		        left_join(b_i, by = 'movieId') %>%
		        left_join(b_u, by = 'userId') %>%
        		mutate(pred = mu_hat + b_i + b_u) %>%
        		pull(pred)
return(RMSE(predicted_ratings, test_set$rating))
})

# Drawing Plot to shows range of lambdas Vs RMSE
lambda_data <- data.frame(lambdas, rmses)
ggplot(data = lambda_data) +  
  	geom_point(aes(x = lambdas, y = rmses))

# Finding the RMSE
rmse_regularisation <- min(rmses) 
rmse_regularisation

# Finding the lambda for minimum RMSE
lambda <-lambdas[which.min(rmses)] 
lambda

# RMSE result:
results <- bind_rows(results, tibble(Method = 'Model 4: Regularised movie and user effects', RMSE = min(rmses)))
results %>% knitr::kable()

# Validation of preferrd model using final_holdout_test set
# Comparing the RMSE obtained in "Regularised, Movie and User Effects" Model algorithm versus ‘Simply the mean’ RMSE

# Prediction based on Mean Rating on the final_holdout_test set
mu_hat <- mean(edx$rating)
mu_hat

# Finding the RMSE
final_model1_rmse <- RMSE(final_holdout_test$rating, mu_hat)
final_model1_rmse

# Saving the results
final_rmse_results = tibble(Method = "Simply the mean", RMSE = final_model1_rmse)
final_rmse_results %>% knitr::kable()

# Predicting via regularisation, movie and user effect model
# Using the minimum lambda value from the earlier models on the final_holdout_test set

min_lambda <- lambda 
mu <- mean(edx$rating)
mu
b_i <- edx %>% group_by(movieId) %>% summarise(b_i = sum(rating - mu) / (n() + min_lambda))
b_u <- edx %>% left_join(b_i, by = "movieId") %>% group_by(userId) %>% summarise(b_u = sum(rating - b_i - mu) / (n() + min_lambda))

# Prediction and Finding the RMSE
predicted_ratings <- final_holdout_test %>% left_join(b_i, by = "movieId") %>% left_join(b_u, by = "userId") %>% mutate(pred = mu + b_i + b_u) %>% pull(pred)
final_rmse_model <- RMSE(final_holdout_test$rating, predicted_ratings)
final_rmse_model

# Saving the results in Data Frame
final_rmse_results <- bind_rows(final_rmse_results, tibble(Method = "Regularised movie and user effects", RMSE = final_rmse_model))
final_rmse_results %>% knitr::kable()

