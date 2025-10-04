#!/usr/bin/env Rscript

suppressMessages(library(optparse))
suppressMessages(library(frbs))

option_list <- list(
  make_option(c("--csv"),    type="character", help="Path to surrogate_dataset.csv", metavar="FILE"),
  make_option(c("--outdir"), type="character", help="Output directory", metavar="DIR"),
  make_option(c("--folds"),  type="integer", default=5, help="CV folds [default %default]"),
  make_option(c("--seed"),   type="integer", default=42, help="Random seed [default %default]"),
  make_option(c("--popu.size"),     type="integer", default=30, help="Population size [default %default]"),
  make_option(c("--max.gen"),       type="integer", default=100, help="Max generations [default %default]"),
  make_option(c("--persen_cross"),  type="double",  default=0.9, help="Crossover prob [default %default]"),
  make_option(c("--persen_mutant"), type="double",  default=0.3, help="Mutation prob [default %default]"),
  make_option(c("--max.num.rule"),  type="integer", default=0,   help="Max number of rules (0 = package default)"),
  make_option(c("--p.dcare"),       type="double",  default=0.5, help="p.dcare [default %default]"),
  make_option(c("--p.gccl"),        type="double",  default=1.0, help="p.gccl [default %default]"),
  make_option(c("--num.class"),     type="integer", default=2,   help="Number of classes [default %default]"),
  make_option(c("--num.labels"),    type="integer", default=3,   help="Fuzzy labels per input [default %default]"),
  make_option(c("--max.iter"),      type="integer", default=NA,  help="Alias of max.gen"),
  make_option(c("--percent.genes"), type="double",  default=NA,  help="Legacy knob"),
  make_option(c("--save_rules"),    action="store_true", default=TRUE,  help="Save frbs_rules.csv"),
  make_option(c("--save_model"),    action="store_true", default=TRUE,  help="Save frbs_model.RData")
)
opt <- parse_args(OptionParser(option_list=option_list))
if (is.null(opt$csv) || is.null(opt$outdir)) {
  stop("Usage: train_fuzzy_surrogate.R --csv file --outdir dir")
}
csv_path <- opt$csv
out_dir  <- opt$outdir
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ---- Load & clean ----
data <- read.csv(csv_path, header=TRUE, check.names=FALSE)
if (!"class" %in% names(data)) stop("CSV must have 'class' column")

X <- data[, setdiff(colnames(data), "class"), drop=FALSE]
Y <- data$class

X[] <- lapply(X, function(col) as.numeric(as.character(col)))
Y   <- as.numeric(as.character(Y))
if (any(is.na(Y))) stop("Non-numeric labels in 'class'.")

ok <- apply(cbind(X, Y), 1, function(r) all(is.finite(r)))
X <- X[ok, , drop=FALSE]
Y <- Y[ok]

non_const <- sapply(X, function(col) length(unique(col)) > 1)
if (!any(non_const)) stop("All input columns became constant.")
X <- X[, non_const, drop=FALSE]

if (length(unique(Y)) < opt$num.class) {
  fake_label <- ifelse(unique(Y)[1] == 0, 1, 0)
  jit <- as.numeric(X[1, , drop=FALSE]) + rnorm(ncol(X), sd=1e-3)
  X <- rbind(X, jit)
  Y <- c(Y, fake_label)
}
if (min(Y, na.rm=TRUE) == 0) Y <- Y + 1L

Xmat <- as.matrix(X)
colnames(Xmat) <- make.names(colnames(Xmat), unique=TRUE)
range.data <- apply(Xmat, 2, range)

# ---- Folds ----
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
folds <- make_folds(Y, k=opt$folds, seed=opt$seed)

# ---- Control ----
num_labels_vec <- rep(as.integer(opt$num.labels), ncol(Xmat))
control <- list(
  num.labels    = num_labels_vec,
  popu.size     = as.integer(opt$popu.size),
  max.gen       = if (!is.na(opt$max.iter)) as.integer(opt$max.iter) else as.integer(opt$max.gen),
  persen_cross  = as.numeric(opt$persen_cross),
  persen_mutant = as.numeric(opt$persen_mutant),
  max.num.rule  = as.integer(opt$max.num.rule),
  p.dcare       = as.numeric(opt$p.dcare),
  p.gccl        = as.numeric(opt$p.gccl),
  name          = "fhgbml-model"
)
if (!is.null(opt$num.class) && !is.na(opt$num.class)) control$num.class <- as.integer(opt$num.class)
if (!is.na(opt$percent.genes)) control$percent.genes <- as.numeric(opt$percent.genes)

# ---- CV ----
metrics_list <- list()
pred_full <- rep(NA_integer_, length(Y))

for (i in seq_along(folds)) {
  test_idx  <- folds[[i]]
  if (length(test_idx) == 0) next
  train_idx <- setdiff(seq_along(Y), test_idx)

  Xt <- Xmat[train_idx, , drop=FALSE]
  Yt <- Y[train_idx]
  Xv <- Xmat[test_idx, , drop=FALSE]
  Yv <- Y[test_idx]

  if (length(unique(Yt)) < 2) {
    fake_label <- ifelse(unique(Yt) == 1, 2, 1)
    jit <- Xt[1, ] + rnorm(ncol(Xt), sd=1e-3)
    Xt <- rbind(Xt, jit); Yt <- c(Yt, fake_label)
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
    Recall=round(rec,4), F1_Score=round(f1,4)
  )
  pred_full[test_idx] <- y_predb
}

# Save CV metrics (even if empty)
if (length(metrics_list) > 0) {
  metrics_df <- do.call(rbind, metrics_list)
} else {
  metrics_df <- data.frame(Fold=integer(0), Accuracy=double(0), Precision=double(0), Recall=double(0), F1_Score=double(0))
}
write.csv(metrics_df, file.path(out_dir, "frbs_cv_metrics_per_fold.csv"), row.names=FALSE)

# ---- Final model on all data ----
if (length(unique(Y)) < 2) {
  fake_label <- ifelse(unique(Y) == 1, 2, 1)
  jit <- Xmat[1, ] + rnorm(ncol(Xmat), sd=1e-3)
  Xmat <- rbind(Xmat, jit); Y <- c(Y, fake_label)
}
final_model <- frbs.learn(
  data.train  = cbind(Xmat, class=Y),
  range.data  = range.data,
  method.type = "FH.GBML",
  control     = control
)
if (isTRUE(opt$save_model)) {
  save(final_model, file=file.path(out_dir, "frbs_model.RData"))
}
if (isTRUE(opt$save_rules) && !is.null(final_model$rule)) {
  rules_df <- as.data.frame(final_model$rule)
  write.csv(rules_df, file.path(out_dir, "frbs_rules.csv"), row.names=FALSE)
}

# Final predictions (for fallback & diagnostics)
final_pred <- predict(final_model, Xmat)
final_pred_bin <- ifelse(final_pred == 2, 1L, 0L)
write.csv(data.frame(Row=seq_along(final_pred_bin), FRBS_Final_Pred=final_pred_bin),
          file.path(out_dir, "frbs_final_predictions.csv"), row.names=FALSE)

# ---- Always produce CV-aligned predictions file ----
# If any CV rows are NA or file might be missing, fill with final model preds.
pred_filled <- pred_full
na_idx <- which(is.na(pred_filled))
if (length(na_idx) > 0) {
  pred_filled[na_idx] <- final_pred_bin[na_idx]
}
write.csv(data.frame(Row=seq_along(pred_filled), FRBS_Pred=as.integer(pred_filled)),
          file.path(out_dir, "frbs_cv_predictions_aligned.csv"), row.names=FALSE)

# Interpretability summary
num_rules <- ifelse(is.null(final_model$rule), NA_integer_, nrow(as.data.frame(final_model$rule)))
mean_rule_len <- NA_real_
write.csv(
  data.frame(
    NumRules=num_rules,
    MeanRuleLength=mean_rule_len,
    PopuSize=control$popu.size,
    MaxGen=control$max.gen,
    NumLabelsPerVar=length(control$num.labels)
  ),
  file.path(out_dir, "frbs_interpret.csv"),
  row.names=FALSE
)

# Control dump
ctrl_dump <- capture.output(str(control))
writeLines(ctrl_dump, con=file.path(out_dir, "frbs_control.txt"))

cat("✅ FRBS (FH.GBML) trained. Outputs saved in:", out_dir, "\n")
