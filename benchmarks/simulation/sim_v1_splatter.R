# Benchmark 1, version 1: Splatter (cell types + technical batches) + one condition effect with known variance shares.
# Condition glue (the only non-Splatter step): counts = Poisson(CellMeans * exp(beta_g * stim_i)), i.e. exactly Splatter's final
# Poisson draw from the BCV-adjusted means, with a known log-fold-change per gene. Truth shares are on the log-mean scale
# (structured part only): celltype = log Base - log Batch, batch = log Batch, condition = beta_g * stim.
args <- commandArgs(TRUE)   # Rscript sim_v1_splatter.R <seed> <batch_scale> <cond_sd> <tag> <out_dir>
seed <- as.integer(args[1]); batch_scale <- as.numeric(args[2]); cond_sd <- as.numeric(args[3]); tag <- args[4]
out <- if (length(args) >= 5) args[5] else "results/sim"; dir.create(out, showWarnings = FALSE, recursive = TRUE)
suppressMessages({library(splatter); library(SingleCellExperiment); library(Matrix)})
set.seed(seed)
G <- 3000; B <- 8; per <- 3000
p <- newSplatParams(nGenes = G, batchCells = rep(per, B), seed = seed)
p <- setParams(p, group.prob = c(.35, .25, .2, .12, .08), de.prob = 0.3, de.facLoc = 0.3, de.facScale = 0.5,
               batch.facLoc = batch_scale, batch.facScale = batch_scale)
s <- splatSimulate(p, method = "groups", verbose = FALSE)
cm <- assay(s, "CellMeans"); bt <- log(assay(s, "BatchCellMeans")); base <- log(assay(s, "BaseCellMeans"))
n <- ncol(cm); batch <- as.integer(factor(s$Batch)); ct <- s$Group
stim <- as.integer(ave(seq_len(n), batch, FUN = function(i) sample(rep(0:1, length.out = length(i)))))   # balanced within batch
affected <- runif(G) < 0.2; beta <- ifelse(affected, rnorm(G, 0, cond_sd), 0)                              # known log fold change per gene
X <- matrix(rpois(length(cm), cm * exp(outer(beta, stim))), nrow = G)
dimnames(X) <- list(rownames(cm), colnames(cm))
v <- function(M) apply(M, 1, var)
vb <- v(bt); vc <- v(base - bt); vd <- v(outer(beta, stim))
keep <- rowMeans(X > 0) >= 0.05
tot <- vb + vc + vd
tr <- data.frame(gene = rownames(cm), var_celltype = vc, var_condition = vd, var_batch = vb,
                 share_celltype = vc / tot, share_condition = vd / tot, share_batch = vb / tot, beta = beta, expressed = keep)
write.csv(tr, file.path(out, paste0(tag, "_truth_gene.csv")), row.names = FALSE)
writeMM(as(Matrix(X, sparse = TRUE), "CsparseMatrix"), file.path(out, paste0(tag, "_counts.mtx")))
write.csv(data.frame(cell = colnames(cm), cell_type = paste0("type", ct), batch = paste0("b", batch), condition = ifelse(stim == 1, "stim", "ctrl")),
          file.path(out, paste0(tag, "_obs.csv")), row.names = FALSE)
writeLines(rownames(cm), file.path(out, paste0(tag, "_genes.txt")))
k <- tr[tr$expressed, ]
cat(sprintf("%s seed %d: cells %d genes %d expressed %d zeros %.2f\n", tag, seed, n, G, nrow(k), mean(X == 0)))
cat(sprintf("GLOBAL variance shares (sum over expressed genes): celltype %.3f condition %.3f batch %.3f\n",
    sum(k$var_celltype) / sum(k$var_celltype + k$var_condition + k$var_batch), sum(k$var_condition) / sum(k$var_celltype + k$var_condition + k$var_batch), sum(k$var_batch) / sum(k$var_celltype + k$var_condition + k$var_batch)))
cat(sprintf("median per-gene shares (expressed): celltype %.3f condition %.3f batch %.3f | mean var: ct %.3f cond %.3f batch %.3f\n",
    median(k$share_celltype), median(k$share_condition), median(k$share_batch), mean(k$var_celltype), mean(k$var_condition), mean(k$var_batch)))
