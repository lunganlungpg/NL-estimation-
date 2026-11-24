test_that("the workflow runs on every bundled dataset", {
  idx <- nl_datasets()
  expect_equal(nrow(idx), 11)
  for (i in seq_len(nrow(idx))) {
    if (idx$dataset[i] == "mgh09") next
    wf <- nl_workflow(get(idx$dataset[i]), idx$x[i], idx$y[i], plot = FALSE)
    expect_s3_class(wf, "nl_workflow")
    expect_true(wf$best$converged, info = idx$dataset[i])
    expect_equal(wf$comparison$model[1], wf$best_model)
    expect_equal(sum(wf$comparison$weight, na.rm = TRUE), 1, tolerance = 1e-8)
  }
})

test_that("the workflow picks the generating model on clear-cut data", {
  expect_equal(nl_workflow(plant_growth, "day", "biomass", plot = FALSE)$best_model, "gompertz")
  expect_equal(nl_workflow(puromycin, "conc", "rate", plot = FALSE)$best_model, "michaelis_menten")
  expect_equal(nl_workflow(dose_response, "conc", "viability", plot = FALSE)$best_model, "four_pl")
})

test_that("the workflow writes tables, a figure and a report", {
  out <- file.path(tempdir(), "nlestim-test-output")
  unlink(out, recursive = TRUE)
  wf <- nl_workflow(crop_yield_nitrogen, "nitrogen", "yield", output_dir = out, plot = FALSE)
  expect_true(all(file.exists(wf$files)))
  expect_setequal(basename(wf$files), c("model_comparison.csv", "best_model_parameters.csv",
                                        "all_model_parameters.csv", "fitted_values.csv",
                                        "prediction_curve.csv", "workflow_plots.png", "report.txt"))
  pars <- read.csv(file.path(out, "best_model_parameters.csv"))
  expect_true(all(c("estimate", "std_error", "lower", "upper") %in% names(pars)))
  expect_output(print(wf), "Best model")
})

test_that("the workflow accepts models, weights by column, and other options", {
  d <- transform(drug_elimination, w = 1 / conc^2)
  wf <- nl_workflow(d, "time", "conc", models = c("exponential_decay", "power_law"),
                    weights = "w", ci_method = "wald", plot = FALSE)
  expect_setequal(wf$comparison$model, c("exponential_decay", "power_law"))
  expect_error(nl_workflow(d, "time", "nope"), "not found")
  expect_error(nl_workflow(d, "time", "conc", models = "nope"), "no model could be fitted")
})

test_that("model suggestions follow the shape of the data", {
  x <- 1:20
  expect_true("michaelis_menten" %in% nl_suggest_models(x, 10 * x / (5 + x)))
  expect_true("exponential_decay" %in% nl_suggest_models(x, exp(-0.2 * x)))
  expect_equal(nl_suggest_models(x, exp(-((x - 10) / 3)^2)), "gaussian_peak")
  expect_false("michaelis_menten" %in% nl_suggest_models(x - 10, x))
  expect_s3_class(nl_models(), "data.frame")
})

test_that("nl_compare ranks models by AICc", {
  tab <- nl_compare(logistic = nl_fit(biomass ~ day, plant_growth, model = "logistic"),
                    gompertz = nl_fit(biomass ~ day, plant_growth, model = "gompertz"))
  expect_equal(tab$model[1], "gompertz")
  expect_true(all(c("AICc", "delta_AICc", "weight", "AIC", "BIC") %in% names(tab)))
})

test_that("nl_compare_solvers reports each method", {
  cmp <- nl_compare_solvers(bod ~ days, boxbod, model = "exponential_rise",
                            start = c(a = 1, k = 1), plot = FALSE)
  expect_equal(cmp$table$method, c("gn-pure", "gn", "lm"))
  expect_equal(cmp$table$converged, c(FALSE, TRUE, FALSE))
  pdf(NULL)
  on.exit(dev.off())
  nl_compare_solvers(rate ~ conc, puromycin, model = "michaelis_menten", plot = TRUE)
})

test_that("nl_read reads CSV and Excel files", {
  csv <- nl_read(system.file("extdata", "puromycin.csv", package = "nlestim"))
  expect_equal(nrow(csv), 12)
  skip_if_not_installed("readxl")
  xl <- nl_read(system.file("extdata", "nonlinear_datasets.xlsx", package = "nlestim"),
                sheet = "plant_growth")
  expect_equal(nrow(xl), 15)
})
