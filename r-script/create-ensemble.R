#
# Description: Sample script for creating ensemble model / prediction
#
# Author: Oliver Muellerklein
#
# IF IT IS NOT OBVIOUS - PLEASE FEEL FREE TO DO WHATEVER YOU WANT WITH THIS
# CODE! EDIT IT - COPY IT - LEARN FROM IT - WHATEVER - Have fun!
#
################################################################################
#### 1. Sample run with sample tweet sentiment data
################################################################################
#### set of 8 models: 2 linear-SVM, 2 kernel-SVM, 3 AUC-XGBoost, 1 Gini-XGBoost
#
# i. Create / Load threshold model
tmpEns.full <- read.csv("data/trial-ens-data/trial_ens_pred.csv")
tmpEns.actualNeg <- tmpEns.full[tmpEns.full$yactual == 0, ]
tmpEns.actualPos <- tmpEns.full[tmpEns.full$yactual == 1, ]

yActual <- tmpEns.full[, 9]

minSVM.neg.mean <- min(mean(tmpEns.actualNeg[, 1]), mean(tmpEns.actualNeg[, 2]), mean(tmpEns.actualNeg[, 3]), mean(tmpEns.actualNeg[, 4]))
# [1] 0.6613333

minXGB.neg.mean <- min(mean(tmpEns.actualNeg[, 5]), mean(tmpEns.actualNeg[, 6]), mean(tmpEns.actualNeg[, 7]), mean(tmpEns.actualNeg[, 8]))
# [1] 0.1905041

minSVM.pos.mean <- min(mean(tmpEns.actualPos[, 1]), mean(tmpEns.actualPos[, 2]), mean(tmpEns.actualPos[, 3]), mean(tmpEns.actualPos[, 4]))
# [1] 0.08863008

minXGB.pos.mean <- min(mean(tmpEns.actualPos[, 5]), mean(tmpEns.actualPos[, 6]), mean(tmpEns.actualPos[, 7]), mean(tmpEns.actualPos[, 8]))
# [1] 0.3206343

### Medians (note: SVMs should be 1 - PROB) - - > they are flipped?!

minSVM.neg.med <- min(median(tmpEns.actualNeg[, 1]), median(tmpEns.actualNeg[, 2]), median(tmpEns.actualNeg[, 3]), median(tmpEns.actualNeg[, 4]))
# [1] 1

minXGB.neg.med <- min(median(tmpEns.actualNeg[, 5]), median(tmpEns.actualNeg[, 6]), median(tmpEns.actualNeg[, 7]), median(tmpEns.actualNeg[, 8]))
# [1] 0

minSVM.pos.med <- min(median(tmpEns.actualPos[, 1]), median(tmpEns.actualPos[, 2]), median(tmpEns.actualPos[, 3]), median(tmpEns.actualPos[, 4]))
# [1] 0

minXGB.pos.med <- min(median(tmpEns.actualPos[, 5]), median(tmpEns.actualPos[, 6]), median(tmpEns.actualPos[, 7]), median(tmpEns.actualPos[, 8]))
# [1] 0

################################################################################
#### 2. Find thresholds
################################################################################

# flip SVM direction (1 - PRED)
lSVM01 <- as.numeric(tmpEns.full$yhat_lin_svm0.1 < minSVM.neg.mean)
kSVM01 <- as.numeric(tmpEns.full$yhat_rad_svm0.1 < minSVM.neg.mean)
lSVM1 <- as.numeric(tmpEns.full$yhat_lin_svm1 < minSVM.neg.mean)
kSVM1 <- as.numeric(tmpEns.full$yhat_rad_svm1 < minSVM.neg.mean)

# do XGB
xgb1 <- as.numeric(tmpEns.full$yhat_xgb1 > minXGB.pos.mean)
xgb2 <- as.numeric(tmpEns.full$yhat_xgb2 > minXGB.pos.mean)
xgb3 <- as.numeric(tmpEns.full$yhat_xgb3 > minXGB.pos.mean)
xgbGini <- as.numeric(tmpEns.full$yhat_xgbGini > minXGB.pos.mean)

################################################################################
#### 3. Create and export data frames
################################################################################

tEnsPred.binary <- data.frame(lSVM01, kSVM01, lSVM1, kSVM1, xgb1, xgb2, xgb3, xgbGini)
tEnsPred.binary.wActual <- data.frame(lSVM01, kSVM01, lSVM1, kSVM1, xgb1, xgb2, xgb3, xgbGini, yActual = tmpEns.full$yactual)

# do complete ensemble pred - mean
svmPred <- data.frame(svmMean = rowMeans(tEnsPred.binary[, 1:4]))
svmPred.wActual <- data.frame(svmMean = rowMeans(tEnsPred.binary[, 1:4]), yActual = yActual)

xgbPred <- data.frame(xgbMean = rowMeans(tEnsPred.binary[, 5:8]))
xgbPred.wActual <- data.frame(xgbMean = rowMeans(tEnsPred.binary[, 5:8]), yActual = yActual)

allEnsPred <- data.frame(allMean = rowMeans(tEnsPred.binary[, 1:8]))
allEnsPred.wActual <- cbind(allEnsPred, yActual = yActual)

# Complete ensemble pred - median

svmPred.med <- data.frame(svmMed = apply(tEnsPred.binary[, 1:4], MARGIN = 1, FUN = median, na.rm = TRUE))
svmPred.med.wActual <- data.frame(svmMed = apply(tEnsPred.binary[, 1:4], MARGIN = 1, FUN = median, na.rm = TRUE), yActual = yActual)

xgbPred.med <- data.frame(xgbMed = apply(tEnsPred.binary[, 5:8], MARGIN = 1, FUN = median, na.rm = TRUE))
xgbPred.med.wActual <- data.frame(xgbMed = apply(tEnsPred.binary[, 5:8], MARGIN = 1, FUN = median, na.rm = TRUE), yActual = yActual)

allEnsPred.med <- data.frame(allMed = apply(tEnsPred.binary[, 1:8], MARGIN = 1, FUN = median, na.rm = TRUE))
allEnsPred.med.wActual <- data.frame(allMed = apply(tEnsPred.binary[, 1:8], MARGIN = 1, FUN = median, na.rm = TRUE), yActual = yActual)

# Export

write.csv(svmPred.wActual, file = "data/trial-ens-data/trial_svm_mean_ens_pred.csv")
write.csv(svmPred.med.wActual, file = "data/trial-ens-data/trial_svm_med_ens_pred.csv")

write.csv(xgbPred.wActual, file = "data/trial-ens-data/trial_xgb_mean_ens_pred.csv")
write.csv(xgbPred.med.wActual, file = "data/trial-ens-data/trial_xgb_med_ens_pred.csv")

write.csv(allEnsPred.wActual, file = "data/trial-ens-data/trial_all_mean_ens_pred.csv")
write.csv(allEnsPred.med.wActual, file = "data/trial-ens-data/trial_all_med_ens_pred.csv")

################################################################################
#### 4. Get accuracy / misclassification
################################################################################

# correct classification (for binary response)
svmMed.correctPred <- (svmPred.med.wActual[, 1] == svmPred.med.wActual[, 2])
svmMed.correctPred <- svmMed.correctPred * 1
svmMed.numCorrect <- sum(svmMed.correctPred)
svmMed.accur <- svmMed.numCorrect / length(yActual)
# [1] 0.7798057

# misclassification error
svmMed.misClass <- (1 - svmMed.accur) * length(yActual)
# [1] 6732
svmMed.misClass / length(yActual)
# [1] 0.2201943

# Compare against MEAN

svmMean.correctPred <- (svmPred.wActual[, 1] == svmPred.wActual[, 2])
svmMean.correctPred <- svmMean.correctPred * 1
svmMean.numCorrect <- sum(svmMean.correctPred)
svmMean.accur <- svmMean.numCorrect / length(yActual)
# [1] 0.7788245

# misclassification error
svmMean.misClass <- (1 - svmMean.accur) * length(yActual)
# [1] 6762
svmMean.misClass / length(yActual)
# [1] 0.2211755

## XGBOOST

xgbMed.correctPred <- (xgbPred.med.wActual[, 1] == xgbPred.med.wActual[, 2])
xgbMed.correctPred <- xgbMed.correctPred * 1
xgbMed.numCorrect <- sum(xgbMed.correctPred)
xgbMed.accur <- xgbMed.numCorrect / length(yActual)
# [1] 0.5594479

# misclassification error
xgbMed.misClass <- (1 - xgbMed.accur) * length(yActual)
# [1] 13469
xgbMed.misClass / length(yActual)
# [1] 0.4405521


# mean xgb

xgbMean.correctPred <- (xgbPred.wActual[, 1] == xgbPred.wActual[, 2])
xgbMean.correctPred <- xgbMean.correctPred * 1
xgbMean.numCorrect <- sum(xgbMean.correctPred)
xgbMean.accur <- xgbMed.numCorrect / length(yActual)
# [1] 0.5594479

# misclassification error
xgbMean.misClass <- (1 - xgbMean.accur) * length(yActual)
# [1] 13469
xgbMean.misClass / length(yActual)
# [1] 0.4405521 

################################################################################
#### END
################################################################################
