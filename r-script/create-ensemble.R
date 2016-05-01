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

### Do thresholding

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

tEnsPred.binary <- data.frame(lSVM01, kSVM01, lSVM1, kSVM1, xgb1, xgb2, xgb3, xgbGini)

tEnsPred.binary.wActual <- data.frame(lSVM01, kSVM01, lSVM1, kSVM1, xgb1, xgb2, xgb3, xgbGini, yActual = tmpEns.full$yactual)

# do complete ensemble pred - mean
svmPred <- data.frame(svmMean = rowMeans(tEnsPred.binary[, 1:4]))

xgbPred <- data.frame(xgbMean = rowMeans(tEnsPred.binary[, 5:8]))

allEnsPred <- data.frame(allMean = rowMeans(tEnsPred.binary[, 1:8]))

allEnsPred.wActual <- cbind(allEnsPred, yActual = yActual)

# Complete ensemble pred - median

svmPred.med <- data.frame(svmMed = apply(tEnsPred.binary[, 1:4], MARGIN = 1, FUN = median, na.rm = TRUE))

xgbPred.med <- data.frame(xgbMed = apply(tEnsPred.binary[, 5:8], MARGIN = 1, FUN = median, na.rm = TRUE))

allEnsPred.med <- data.frame(allMed = apply(tEnsPred.binary[, 1:8], MARGIN = 1, FUN = median, na.rm = TRUE))

allEnsPred.med.wActual <- cbind(allEnsPred.med, yActual = yActual)

################################################################################
#### 2.
################################################################################





################################################################################
#### 3.
################################################################################






################################################################################
#### END
################################################################################
