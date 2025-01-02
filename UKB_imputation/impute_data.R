### Script to impute missing data
### 
### Author: Angelos Manolias
### date: 19/11/2024
### 
### Load libraries
library(missForest)
library(tidyverse)
library(randomForest)
library(foreach)
library(rngtools)
library(doParallel)
library(unix)

rlimit_as(16 * 1024^3)
print(rlimit_as())

args <- commandArgs(trailingOnly = T)

### Set random seed
random_seed <- as.numeric(args[2])
set.seed(random_seed)

# Get path
path=dirname(args[1])

# Get output file names
output_file=gsub(".csv","_imputed.csv",args[1])
output_data=gsub(".csv","_imputed.RData",args[1])

### Load data and run imputation
input_data <- read.csv(args[1])

registerDoParallel(cores = 15)

input_data$age <- as.factor(input_data$age)
input_data$sex <- as.factor(input_data$sex)

imp.data <- input_data %>% column_to_rownames("ID")

m.forest <- missForest(imp.data, verbose = T, replace = F, parallelize = "forests",
                       ntree = 50)
save(m.forest, file = output_data)
write.csv(m.forest$ximp, output_file)
m.forest$OOBerror
