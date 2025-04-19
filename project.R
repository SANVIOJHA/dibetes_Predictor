# Install necessary libraries if not already installed
install.packages(c("mlbench", "class", "e1071", "rpart", "neuralnet", "ggplot2", "caret"))

# Load libraries
library(mlbench)
library(class)
library(e1071)
library(rpart)
library(neuralnet)
library(ggplot2)
library(caret)

# Load the dataset
data("PimaIndiansDiabetes")
df <- PimaIndiansDiabetes

# Data Preprocessing
# Median Imputation for Numeric Columns Only
numeric_cols <- sapply(df, is.numeric)
df[numeric_cols] <- lapply(df[numeric_cols], function(x) { x[is.na(x)] <- median(x, na.rm = TRUE); x })

# Split the data into training and testing sets (80-20 split)
set.seed(123)
sample_index <- sample(1:nrow(df), 0.8 * nrow(df))
train_data <- df[sample_index, ]
test_data <- df[-sample_index, ]

# Normalize data for KNN
normalize <- function(x) { return ((x - min(x)) / (max(x) - min(x))) }
train_data_norm <- as.data.frame(lapply(train_data[,1:8], normalize))
test_data_norm <- as.data.frame(lapply(test_data[,1:8], normalize))
train_data_norm$diabetes <- train_data$diabetes

# Train models

## K-Nearest Neighbors (KNN)
knn_pred <- knn(train_data_norm[,1:8], test_data_norm[,1:8], train_data_norm$diabetes, k=5)

## Naive Bayes
nb_model <- naiveBayes(diabetes ~ ., data=train_data)
nb_pred <- predict(nb_model, test_data)

## Decision Tree
dt_model <- rpart(diabetes ~ ., data=train_data, method="class")
dt_pred <- predict(dt_model, test_data, type="class")

## Neural Network
scaled_train <- as.data.frame(scale(train_data[,1:8]))
scaled_test <- as.data.frame(scale(test_data[,1:8]))
scaled_train$diabetes <- ifelse(train_data$diabetes == "pos", 1, 0)
nn_model <- neuralnet(diabetes ~ ., data=scaled_train, hidden=5, linear.output=FALSE)
nn_pred <- compute(nn_model, scaled_test)$net.result
nn_pred <- ifelse(nn_pred > 0.5, "pos", "neg")

## Support Vector Machine (SVM)
svm_model <- svm(diabetes ~ ., data=train_data, kernel="radial")
svm_pred <- predict(svm_model, test_data)

# Combine predictions for voting ensemble
combined_preds <- data.frame(KNN = knn_pred, NaiveBayes = nb_pred, 
                             DecisionTree = dt_pred, NeuralNet = nn_pred, 
                             SVM = svm_pred)
vote_pred <- apply(combined_preds, 1, function(row) {
  ifelse(mean(row == "pos") > 0.5, "pos", "neg")
})

# Calculate accuracy for each model and ensemble
knn_accuracy <- sum(test_data$diabetes == knn_pred) / nrow(test_data)
nb_accuracy <- sum(test_data$diabetes == nb_pred) / nrow(test_data)
dt_accuracy <- sum(test_data$diabetes == dt_pred) / nrow(test_data)
nn_accuracy <- sum(test_data$diabetes == nn_pred) / nrow(test_data)
svm_accuracy <- sum(test_data$diabetes == svm_pred) / nrow(test_data)
vote_accuracy <- sum(test_data$diabetes == vote_pred) / nrow(test_data)

# Create a data frame for accuracies
accuracies <- data.frame(
  Model = c("KNN", "Naive Bayes", "Decision Tree", "Neural Network", "SVM", "Voting Ensemble"),
  Accuracy = c(knn_accuracy, nb_accuracy, dt_accuracy, nn_accuracy, svm_accuracy, vote_accuracy)
)

# Find the best performing model
best_model <- accuracies[which.max(accuracies$Accuracy), ]

# Visualization for model accuracies
ggplot(accuracies, aes(x = Model, y = Accuracy)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "Model Accuracies", x = "Model", y = "Accuracy") +
  theme_minimal()

# Confusion Matrix Plotting Function
plot_confusion_matrix <- function(predictions, true_labels, model_name) {
  cm <- confusionMatrix(factor(predictions, levels=c("neg", "pos")), 
                        factor(true_labels, levels=c("neg", "pos")))
  cm_table <- as.data.frame(cm$table)  # Convert confusion matrix to data frame for ggplot
  ggplot(cm_table, aes(x = Prediction, y = Reference, fill = Freq)) +
    geom_tile(color = "white") +
    scale_fill_gradient(low = "white", high = "steelblue") +
    geom_text(aes(label = Freq), color = "black") +
    labs(title = paste("Confusion Matrix -", model_name), x = "Predicted", y = "Actual") +
    theme_minimal()
}

# Plot Confusion Matrices for each model
plot_confusion_matrix(knn_pred, test_data$diabetes, "KNN")
plot_confusion_matrix(nb_pred, test_data$diabetes, "Naive Bayes")
plot_confusion_matrix(dt_pred, test_data$diabetes, "Decision Tree")
plot_confusion_matrix(nn_pred, test_data$diabetes, "Neural Network")
plot_confusion_matrix(svm_pred, test_data$diabetes, "SVM")
plot_confusion_matrix(vote_pred, test_data$diabetes, "Voting Ensemble")

# UI for Shiny App-like Output
ui <- fluidPage(
  titlePanel("Diabetes Prediction Dashboard"),
  sidebarLayout(
    sidebarPanel(
      h4("Best Performing Model"),
      textOutput("best_model")
    ),
    mainPanel(
      h3("Best Model Accuracy Visualization"),
      plotOutput("accuracy_plot")
    )
  )
)

server <- function(input, output) {
  # Display the best model
  output$best_model <- renderText({
    paste("Best Model: ", best_model$Model, "with accuracy: ", round(best_model$Accuracy, 2))
  })
  
  # Render the accuracy plot
  output$accuracy_plot <- renderPlot({
    ggplot(accuracies, aes(x = Model, y = Accuracy)) +
      geom_bar(stat = "identity", fill = "skyblue") +
      labs(title = "Best Model Accuracy", x = "Model", y = "Accuracy") +
      theme_minimal()
  })
}

# Run the Shiny app
shinyApp(ui = ui, server = server)






#############################################################################################

library(e1071)
library(mlbench)
library(caret)
library(shiny)
library(ggplot2)

# Load and scale the dataset
getwd()
data(PimaIndiansDiabetes)
diabetes_data <- PimaIndiansDiabetes
View(diabetes_data)

diabetes_data[, 1:8] <- scale(diabetes_data[, 1:8])

# Split data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(diabetes_data$diabetes, p = 0.7, list = FALSE)
train_data <- diabetes_data[trainIndex, ]
test_data <- diabetes_data[-trainIndex, ]

# Train the SVM model
svm_model <- svm(diabetes ~ ., data = train_data, kernel = "radial", cost = 1, gamma = 0.1)

# Define the function to predict diabetes status
predict_diabetes <- function(glucose, pressure, insulin, bmi, pedigree, age) {
  # Calculate mean and standard deviation for scaling
  glucose_mean <- mean(PimaIndiansDiabetes$glucose)
  glucose_sd <- sd(PimaIndiansDiabetes$glucose)
  
  pressure_mean <- mean(PimaIndiansDiabetes$pressure)
  pressure_sd <- sd(PimaIndiansDiabetes$pressure)
  
  insulin_mean <- mean(PimaIndiansDiabetes$insulin)
  insulin_sd <- sd(PimaIndiansDiabetes$insulin)
  
  mass_mean <- mean(PimaIndiansDiabetes$mass)
  mass_sd <- sd(PimaIndiansDiabetes$mass)
  
  pedigree_mean <- mean(PimaIndiansDiabetes$pedigree)
  pedigree_sd <- sd(PimaIndiansDiabetes$pedigree)
  
  age_mean <- mean(PimaIndiansDiabetes$age)
  age_sd <- sd(PimaIndiansDiabetes$age)
  
  # Prepare new data row with input values, scaled according to calculated means and SDs
  new_data <- data.frame(
    pregnant = 0,  # Placeholder for pregnancy count
    glucose = (glucose - glucose_mean) / glucose_sd,
    pressure = (pressure - pressure_mean) / pressure_sd,
    triceps = 0,   # Placeholder for triceps skin fold thickness
    insulin = (insulin - insulin_mean) / insulin_sd,
    mass = (bmi - mass_mean) / mass_sd,
    pedigree = (pedigree - pedigree_mean) / pedigree_sd,
    age = (age - age_mean) / age_sd
  )
  
  # Predict using the trained SVM model
  result <- predict(svm_model, new_data)
  return(result)
}

# Define the Shiny app
ui <- fluidPage(
  titlePanel("Diabetes Prediction App"),
  sidebarLayout(
    sidebarPanel(
      numericInput("glucose", "Glucose Level:", value = 120, min = 0),
      numericInput("pressure", "Blood Pressure:", value = 70, min = 0),
      numericInput("insulin", "Insulin Level:", value = 80, min = 0),
      numericInput("bmi", "BMI:", value = 30, min = 0),
      numericInput("pedigree", "Diabetes Pedigree Function:", value = 0.5, min = 0),
      numericInput("age", "Age:", value = 33, min = 0),
      actionButton("predict", "Predict")
    ),
    mainPanel(
      textOutput("result"),
      plotOutput("plot")
    )
  )
)

server <- function(input, output) {
  observeEvent(input$predict, {
    # Get input values
    glucose <- input$glucose
    pressure <- input$pressure
    insulin <- input$insulin
    bmi <- input$bmi
    pedigree <- input$pedigree
    age <- input$age
    
    # Make prediction
    prediction <- predict_diabetes(glucose, pressure, insulin, bmi, pedigree, age)
    
    # Display result
    output$result <- renderText({
      paste("Prediction: ", ifelse(prediction == "pos", "Diabetic", "Not Diabetic"))
    })
    
    # Create a plot to visualize the result
    output$plot <- renderPlot({
      df <- data.frame(Status = c("Diabetic", "Not Diabetic"), Count = c(ifelse(prediction == "pos", 1, 0), ifelse(prediction == "neg", 1, 0)))
      ggplot(df, aes(x = Status, y = Count)) +
        geom_bar(stat = "identity", fill = c("red", "green")) +
        theme_minimal() +
        labs(title = "Diabetes Prediction Result", x = "Status", y = "Count")
    })
  })
}

# Run the app
shinyApp(ui = ui, server = server)


