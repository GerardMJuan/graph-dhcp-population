# Step 1: Load Required Libraries
library(readxl)
library(dplyr)
library(tidyr)
library(purrr)
library(kableExtra) # For nicer table visualization
library(neuroCombat)
install.packages("openxlsx")
library(openxlsx)
# # Read the Excel File
# data <- read_excel("data/multifact_dhcp_merged.xlsx")

# # Exclude non-biomarker columns
# biomarker_columns <- setdiff(colnames(data), c("participant_id", "GRUPO", "center", "gestational_weeks"))

# # Define sites and pairwise comparisons
# sites <- c("dhcp", "multifact", "multifact-healthy")
# site_pairs <- combn(sites, 2, simplify = FALSE)

# # Adjusted function to perform KS test
# perform_ks_test <- function(data, biomarker, pair) {
#   group1_data <- filter(data, center == pair[1])[[biomarker]]
#   group2_data <- filter(data, center == pair[2])[[biomarker]]
  
#   test_result <- ks.test(group1_data, group2_data)
#   p_value <- test_result$p.value
  
#   return(p_value)
# }

# # Adjust the iteration process to ensure strings are handled correctly
# results <- expand.grid(biomarker = biomarker_columns, pair = I(site_pairs)) %>%
#   rowwise() %>%
#   mutate(p_value = list(perform_ks_test(data, biomarker, pair))) %>%
#   unnest(cols = c(p_value)) %>%
#   ungroup() %>%
#   mutate(pair_name = map_chr(pair, ~paste(.x, collapse = " vs "))) %>%
#   select(-pair) %>%
#   pivot_wider(names_from = pair_name, values_from = p_value)

# # Display results
# kable(results) %>%
#   kable_styling(bootstrap_options = c("striped", "hover"))

# Read the Excel File
data <- read_excel("data/multifact_dhcp_merged.xlsx")

# dades
harmonized_data <- data # Initialize harmonized data as a copy of the original

data$center <- as.numeric(factor(data$center))
data$GRUPO <- as.numeric(factor(data$GRUPO))

# Exclude non-biomarker columns
biomarker_columns <- setdiff(colnames(data), c("participant_id", "GRUPO", "center", "gestational_weeks"))

# Extract the biomarker data
biomarker_data <- data[biomarker_columns]

# Transpose the data for ComBat
biomarker_data <- t(biomarker_data)


gestational_weeks = data$gestational_weeks
GRUPO = data$GRUPO

# Prepare the data for ComBat
mod = model.matrix(~ gestational_weeks + GRUPO)
batch = data$center

# Apply ComBat harmonization
combat_data.harmonized <- neuroCombat(dat=biomarker_data,
                            batch=batch,
                            mod=mod)

# transpose the data back
combat_matrix <- t(combat_data.harmonized$dat.combat)

# check size of the data
dim(combat_matrix)
dim(harmonized_data[biomarker_columns])

# Update the harmonized data
harmonized_data[biomarker_columns] <- combat_matrix

# Save the harmonized data to xlsx
write.xlsx(harmonized_data, "data/multifact_dhcp_harmonized.xlsx", rowNames = FALSE)