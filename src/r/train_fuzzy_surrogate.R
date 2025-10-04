#!/usr/bin/env Rscript

suppressMessages(library(optparse))

option_list <- list(
  make_option(c("--csv"), type="character", help="Path to surrogate_dataset.csv", metavar="FILE"),
  make_option(c("--outdir"), type="character", help="Output directory", metavar="DIR")
)
opt <- parse_args(OptionParser(option_list=option_list))

if (is.null(opt$csv) || is.null(opt$outdir)) {
  stop("Usage: train_fuzzy_surrogate.R --csv path/to/surrogate_dataset.csv --outdir out/dir")
}

csv_path  <- opt$csv
out_dir   <- opt$outdir

suppressMessages(library(frbs))

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

data <- read.csv(csv_path, header=TRUE, check.names=FALSE)
if (!"class" %in% names(data)) stop("CSV must have 'class' column")

# Clean and prepare
X <- data[, setdiff(colnames(data), "class"), drop=FALSE]
Y <- data$class
X[] <- lapply(X, function(col) as.numeric(as.character(col)))
Y   <- as.numeric(as.character(Y))
if (any(is.na(Y))) stop("Non-numeric labels in 'class'.")

# Convert 0/1 -> 1/2 for frbs
if (min(Y, na.rm=TRUE) == 0) Y <- Y + 1L

# Remove rows w/ NA/Inf
good <- apply(cbind(X, Y), 1, function(r) all(is.finite(r)))
X <- X[good, , drop=FALSE]
Y <- Y[good]

# Drop constant columns
non_const <- sapply(X, function(col) length(unique(col)) > 1)
X <- X[, non_const, drop=FALSE]

range.data <- apply(X, 2, range)

# Stratified 5-fold
make_folds <- function(y, k=5, seed=42) {
  set.seed(seed)
  y <- as.factor(y)
  idx_by_class <- split(seq_along(y), y)
  folds <- vector("list", k)
  for (ids in idx_by_class) {
    ids <- sample(ids)
    parts <- split(ids, cut(seq_along(ids), breaks=min(k, length(ids)), labels=FALSE))
    for (i in 1:k) {
      if (i <= length(parts) && length(parts[[i]]) > 0) {
        folds[[i]] <- c(folds[[i]], parts[[i]])
      }
    }
  }
  lapply(folds, function(x) sort(unique(x)))
}
folds <- make_folds(Y, k=5, seed=42)

control <- list(
  num.labels    = 3,
  max.iter      = 100,
  popu.size     = 30,
  percent.genes = 0.5,
  name          = "fhgbml-model"
)

metrics_list <- list()
pred_full <- rep(NA_integer_, length(Y))

for (i in seq_along(folds)) {
  test_idx <- folds[[i]]
  if (length(test_idx) == 0) next
  train_idx <- setdiff(seq_along(Y), test_idx)

  Xt <- X[train_idx, , drop=FALSE]
  Yt <- Y[train_idx]
  Xv <- X[test_idx, , drop=FALSE]
  Yv <- Y[test_idx]

  # single-class guard
  if (length(unique(Yt)) < 2) {
    fake_label <- ifelse(unique(Yt) == 1, 2, 1)
    fake_sample <- Xt[1, , drop=FALSE]
    Xt <- rbind(Xt, fake_sample)
    Yt <- c(Yt, fake_label)
  }

  model <- frbs.learn(
    data.train  = cbind(Xt, class=Yt),
    range.data  = range.data,
    method.type = "FH.GBML",
    control     = control
  )

  y_pred  <- predict(model, Xv)
  y_predb <- ifelse(y_pred == 2, 1L, 0L)
  y_trueb <- ifelse(Yv == 2, 1L, 0L)

  TP <- sum(y_predb == 1 & y_trueb == 1)
  TN <- sum(y_predb == 0 & y_trueb == 0)
  FP <- sum(y_predb == 1 & y_trueb == 0)
  FN <- sum(y_predb == 0 & y_trueb == 1)

  acc <- (TP + TN) / length(y_trueb)
  prec <- ifelse((TP + FP) == 0, 0, TP / (TP + FP))
  rec  <- ifelse((TP + FN) == 0, 0, TP / (TP + FN))
  f1   <- ifelse((prec + rec) == 0, 0, 2 * prec * rec / (prec + rec))

  metrics_list[[length(metrics_list)+1]] <- data.frame(
    Fold=i, Accuracy=round(acc,4), Precision=round(prec,4),
    Recall=round(rec,4), F1=round(f1,4)
  )

  pred_full[test_idx] <- y_predb
}

metrics_df <- do.call(rbind, metrics_list)
write.csv(metrics_df, file.path(out_dir, "frbs_cv_metrics_per_fold.csv"), row.names=FALSE)
write.csv(data.frame(Row=seq_along(pred_full), Pred=pred_full),
          file.path(out_dir, "frbs_predictions.csv"), row.names=FALSE)

# Train final model on all data
final_model <- frbs.learn(
  data.train  = cbind(X, class=Y),
  range.data  = range.data,
  method.type = "FH.GBML",
  control     = control
)

save(final_model, file=file.path(out_dir, "frbs_model.RData"))

# Save rules (if available)
if (!is.null(final_model$rule)) {
  rules_df <- as.data.frame(final_model$rule)
  write.csv(rules_df, file.path(out_dir, "frbs_rules.csv"), row.names=FALSE)
}

# Save interpretability metrics
num_rules <- ifelse(is.null(final_model$rule), NA_integer_, nrow(as.data.frame(final_model$rule)))
mean_rule_len <- NA_real_
# (optional simple estimate could be added here if needed)

write.csv(
  data.frame(NumRules=num_rules, MeanRuleLength=mean_rule_len),
  file.path(out_dir, "frbs_interpret.csv"),
  row.names=FALSE
)

cat("✅ FRBS surrogate trained. Outputs written to:", out_dir, "\n")
