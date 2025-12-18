#     Multiclass classification of Bike sharing demand in Seoul       #
# ==================================================================  #
# R script for Exploratory Data Analysis and Predictive Modelling     #
# Data set: SeoulBikeData.csv (8760 rows, 14 columns)                 #
# Author: Christo Tomy                                                #
# R version recommendation: R (>= 4.0.0). The code prints actual      #
# session info at the end.                                            #
# Recommended package versions are printed below when the script runs #
# ==================================================================  #




# ==================================================================  #
#                      Instructions for the user:                     #
# ------------------------------------------------------------------  #
# 1) Place SeoulBikeData.csv in your R working directory, or set      #
# the path in 'data_path' below.                                      #
# 2) Run this script in an R console / RStudio. All outputs will      #
# print to the console and plots will appear in the plotting pane.    #
# The script is verbose and has many comments to help the client.     #
# 3) If a package is missing, install it first (commented install     #
# lines provided).                                                    #
# ==================================================================  #





#######################################################################
###                      Introductory Steps                         ###
#######################################################################
###                       Preliminary Step                          ###
#                                                                     #
# Ensuring your RStudio is up to date will make all code below run    #
# smoothly but will also ensure all packages are usable. Next,        #
# pressing down "CTRL + Enter" will run the line of code you are      #
# currently on or code you have highlighted. I suggest you run the    #
# code a line at a time to ensure R has enough time to properly       #
# understand what it is being asked to do.                            #
#######################################################################



install.packages("tidyverse")
install.packages("janitor")
install.packages("lubridate")
install.packages("skimr")
install.packages("ggcorrplot")
install.packages("caret")
install.packages("nnet")
install.packages("randomForest")
install.packages("keras")
install.packages("tensorflow")

# Packages
library(tidyverse)
library(janitor)
library(lubridate)
library(skimr)
library(ggcorrplot)
library(caret)
library(nnet)
library(randomForest)
library(keras)
library(tensorflow)


# Load data
df_raw <- read.csv("SeoulBikeData.csv", stringsAsFactors = FALSE)

# Clean column names to snake_case (important)
df <- df_raw %>%
  clean_names()

# Quick glance

glimpse(df)
skim(df)

df <- df %>%
  mutate(
    demand_class = case_when(
      rented_bike_count <= quantile(rented_bike_count, 1/3, na.rm = TRUE) ~ "Low",
      rented_bike_count <= quantile(rented_bike_count, 2/3, na.rm = TRUE) ~ "Medium",
      TRUE ~ "High"
    ),
    demand_class = factor(demand_class, levels = c("Low", "Medium", "High"))
  )


# Class Balance, used for classification 

ggplot(df, aes(x = demand_class)) +
  geom_bar() +
  labs(title = "Demand Class Distribution", x = "Demand Class", y = "Count")

# Rentals by hour

ggplot(df, aes(x = hour, y = rented_bike_count)) +
  geom_point(alpha = 0.15) +
  geom_smooth(se = FALSE) +
  labs(title = "Rented Bike Count by Hour", x = "Hour of Day", y = "Rented Bike Count")

# Median by hour 

df %>%
  group_by(hour) %>%
  summarise(median_rentals = median(rented_bike_count, na.rm = TRUE)) %>%
  ggplot(aes(hour, median_rentals)) +
  geom_line() +
  geom_point() +
  labs(title = "Median Rentals by Hour", x = "Hour of Day", y = "Median Rented Bike Count")

# Hour pattern split by class

df %>%
  group_by(hour, demand_class) %>%
  summarise(median_rentals = median(rented_bike_count, na.rm = TRUE), .groups = "drop") %>%
  ggplot(aes(x = hour, y = median_rentals, group = demand_class)) +
  geom_line() +
  geom_point() +
  labs(title = "Median Rentals by Hour and Demand Class",
       x = "Hour of Day", y = "Median Rentals")

# Demand class by season

ggplot(df, aes(x = seasons, fill = demand_class)) +
  geom_bar(position = "fill") +
  labs(title = "Demand Class Proportions by Season",
       x = "Season", y = "Proportion")

# Demand Class by holiday 

ggplot(df, aes(x = holiday, fill = demand_class)) +
  geom_bar(position = "fill") +
  labs(title = "Demand Class Proportions by holiday",
       x = "Holiday", y = "Proportion")

# Key numeric predictors by class

num_vars <- c("temperature_c", "humidity", "wind_speed_m_s", "rainfall_mm", "snowfall_cm")

df_long <- df %>%
  select(all_of(num_vars), demand_class) %>%
  pivot_longer(cols = all_of(num_vars), names_to = "variable", values_to = "value")

ggplot(df_long, aes(x = demand_class, y = value)) +
  geom_boxplot(outlier.alpha = 0.2) +
  facet_wrap(~ variable, scales = "free_y") +
  labs(title = "Key Predictors by Demand Class", x = "Demand Class", y = "Value")

# Correlation heatmap 

num_df <- df %>%
  select(where(is.numeric)) %>%
  select(-rented_bike_count)  # optional: remove target if you want

corr_mat <- cor(num_df, use = "pairwise.complete.obs")

ggcorrplot(corr_mat, lab = FALSE) +
  labs(title = "Correlation Heatmap (Numeric Predictors)")

# Missing data check

miss <- df %>%
  summarise(across(everything(), ~ sum(is.na(.)))) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "missing_n") %>%
  arrange(desc(missing_n))

ggplot(miss, aes(x = reorder(variable, missing_n), y = missing_n)) +
  geom_col() +
  coord_flip() +
  labs(title = "Missing Values by Variable", x = "Variable", y = "Number Missing")


# Keep relevant variables
df_model <- df %>%
  select(
    demand_class,        
    hour,
    temperature_c,
    humidity,
    wind_speed_m_s,
    rainfall_mm,
    snowfall_cm,
    seasons,
    holiday,
    functioning_day
  )


# Converting categorical variables to factors 

df_model <- df_model %>%
  mutate(
    demand_class = factor(demand_class, levels = c("Low", "Medium", "High")),
    seasons = factor(seasons),
    holiday = factor(holiday),
    functioning_day = factor(functioning_day)
  )

# Train - test split 

set.seed(123)

train_index <- createDataPartition(
  df_model$demand_class,
  p = 0.7,
  list = FALSE
)

train_data <- df_model[train_index, ]
test_data  <- df_model[-train_index, ]

prop.table(table(train_data$demand_class))
prop.table(table(test_data$demand_class))

# Scaling numerical variables 

# Identify numeric predictors

num_vars <- c(
  "hour",
  "temperature_c",
  "humidity",
  "wind_speed_m_s",
  "rainfall_mm",
  "snowfall_cm"
)

# Fit scaler on training data only

preproc <- preProcess(
  train_data[, num_vars],
  method = c("center", "scale")
)

# Apply scaling

train_scaled <- train_data
test_scaled  <- test_data

train_scaled[, num_vars] <- predict(preproc, train_data[, num_vars])
test_scaled[, num_vars]  <- predict(preproc, test_data[, num_vars])


# Fit the Multinomial GLM

# Fit multinomial logistic regression

glm_multinom <- multinom(
  demand_class ~ .,
  data = train_scaled
)

# Model summry

summary(glm_multinom)

z_vals <- summary(glm_multinom)$coefficients / summary(glm_multinom)$standard.errors
p_vals <- 2 * (1 - pnorm(abs(z_vals)))

p_vals

# Generate predictions on test data

glm_pred <- predict(
  glm_multinom,
  newdata = test_scaled,
  type = "class"
)

# Confusion matrix


confusionMatrix(
  glm_pred,
  test_scaled$demand_class
)


# Class probabilities 

glm_prob <- predict(
  glm_multinom,
  newdata = test_scaled,
  type = "prob"
)

head(glm_prob)


# Fit the Random Forest Model 


set.seed(123)

rf_model <- randomForest(
  demand_class ~ .,
  data = train_data,
  ntree = 500,
  importance = TRUE
)

# Model Summary

print(rf_model)

# Variable importance plot

varImpPlot(
  rf_model,
  main = "Random Forest Variable Importance"
)

# Predictors on test data

rf_pred <- predict(
  rf_model,
  newdata = test_data,
  type = "class"
)

# Confusion matrix and accuracy

confusionMatrix(
  rf_pred,
  test_data$demand_class
)


rf_prob <- predict(
  rf_model,
  newdata = test_data,
  type = "prob"
)

head(rf_prob)


# Prepare X and Y

# Create design matrices

x_train <- model.matrix(demand_class ~ . , data = train_scaled)[, -1]
x_test  <- model.matrix(demand_class ~ . , data = test_scaled)[, -1]

# Convert target to integer class indices 0:(K-1) then one-hot

y_train_int <- as.integer(train_scaled$demand_class) - 1
y_test_int  <- as.integer(test_scaled$demand_class) - 1

num_classes <- length(levels(train_scaled$demand_class))

y_train <- keras::to_categorical(y_train_int, num_classes = num_classes)
y_test  <- keras::to_categorical(y_test_int,  num_classes = num_classes)


# Build the neural network

set.seed(123)
tf$random$set_seed(123)

nn_model <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu", input_shape = ncol(x_train)) %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = num_classes, activation = "softmax")

nn_model %>% compile(
  optimizer = optimizer_adam(learning_rate = 0.001),
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

summary(nn_model)


# Train with early stopping 

early_stop <- callback_early_stopping(
  monitor = "val_loss",
  patience = 10,
  restore_best_weights = TRUE
)

history <- nn_model %>% fit(
  x = x_train,
  y = y_train,
  epochs = 100,
  batch_size = 32,
  validation_split = 0.2,
  callbacks = list(early_stop),
  verbose = 1
)

plot(history)

# Evaluate on test set 

nn_eval <- nn_model %>% evaluate(x_test, y_test, verbose = 0)
nn_eval

# Predictions and confusion matrix 

# Predicted class probabilities

nn_prob <- nn_model %>% predict(x_test)

# Convert probability to class labels

nn_pred_int <- max.col(nn_prob) - 1

# Convert back to factor with same levels

class_levels <- levels(train_scaled$demand_class)
nn_pred <- factor(class_levels[nn_pred_int + 1], levels = class_levels)

# Confusion matrix

confusionMatrix(nn_pred, test_scaled$demand_class)


