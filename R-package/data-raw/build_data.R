# Builds data/*.rda from the repository's CSV files (the single source of truth
# is nlsfit/datasets.py; scripts/build_data.py exports it to ../data).
# Run from the R-package directory:  Rscript data-raw/build_data.R

read <- function(name, cols) {
  d <- utils::read.csv(file.path("..", "data", paste0(name, ".csv")), check.names = FALSE)
  names(d) <- cols
  d
}
misra1a <- read("misra1a", c("pressure", "volume"))
boxbod <- read("boxbod", c("days", "bod"))
danwood <- read("danwood", c("temperature", "energy"))
rat42 <- read("rat42", c("time", "yield"))
rat43 <- read("rat43", c("time", "weight"))
mgh09 <- read("mgh09", c("x", "y"))
puromycin <- read("puromycin", c("conc", "rate"))
crop_yield_nitrogen <- read("crop_yield_nitrogen", c("nitrogen", "yield"))
plant_growth <- read("plant_growth", c("day", "biomass"))
dose_response <- read("dose_response", c("conc", "viability"))
drug_elimination <- read("drug_elimination", c("time", "conc"))

nist_certified <- data.frame(
  dataset = rep(c("misra1a", "boxbod", "danwood", "rat42", "rat43", "mgh09"), c(2, 2, 2, 3, 4, 4)),
  formula = rep(c("volume ~ b1 * (1 - exp(-b2 * pressure))", "bod ~ b1 * (1 - exp(-b2 * days))",
                  "energy ~ b1 * temperature^b2", "yield ~ b1 / (1 + exp(b2 - b3 * time))",
                  "weight ~ b1 / (1 + exp(b2 - b3 * time))^(1 / b4)",
                  "y ~ b1 * (x^2 + x * b2) / (x^2 + x * b3 + b4)"), c(2, 2, 2, 3, 4, 4)),
  parameter = c("b1", "b2", "b1", "b2", "b1", "b2", "b1", "b2", "b3", "b1", "b2", "b3", "b4",
                "b1", "b2", "b3", "b4"),
  certified = c(2.3894212918e+02, 5.5015643181e-04, 2.1380940889e+02, 5.4723748542e-01,
                7.6886226176e-01, 3.8604055871e+00, 7.2462237576e+01, 2.6180768402e+00,
                6.7359200066e-02, 6.9964151270e+02, 5.2771253025e+00, 7.5962938329e-01,
                1.2792483859e+00, 1.9280693458e-01, 1.9128232873e-01, 1.2305650693e-01,
                1.3606233068e-01),
  start1 = c(500, 1e-4, 1, 1, 1, 5, 100, 1, 0.1, 100, 10, 1, 1, 25, 39, 41.5, 39),
  start2 = c(250, 5e-4, 100, 0.75, 0.7, 4, 75, 2.5, 0.07, 700, 5, 0.75, 1.3, 0.25, 0.39, 0.415, 0.39),
  certified_sse = rep(c(1.2455138894e-01, 1.1680088766e+03, 4.3173084083e-03, 8.0565229338e+00,
                        8.7864049080e+03, 3.0750560385e-04), c(2, 2, 2, 3, 4, 4)),
  stringsAsFactors = FALSE
)

save(misra1a, file = "data/misra1a.rda", compress = "xz")
save(boxbod, file = "data/boxbod.rda", compress = "xz")
save(danwood, file = "data/danwood.rda", compress = "xz")
save(rat42, file = "data/rat42.rda", compress = "xz")
save(rat43, file = "data/rat43.rda", compress = "xz")
save(mgh09, file = "data/mgh09.rda", compress = "xz")
save(puromycin, file = "data/puromycin.rda", compress = "xz")
save(crop_yield_nitrogen, file = "data/crop_yield_nitrogen.rda", compress = "xz")
save(plant_growth, file = "data/plant_growth.rda", compress = "xz")
save(dose_response, file = "data/dose_response.rda", compress = "xz")
save(drug_elimination, file = "data/drug_elimination.rda", compress = "xz")
save(nist_certified, file = "data/nist_certified.rda", compress = "xz")
cat("wrote", length(list.files("data")), "datasets\n")
