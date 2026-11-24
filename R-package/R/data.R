#' NIST Misra1a: dental adsorption (lower difficulty)
#'
#' Monomolecular adsorption data (Misra, 1978). Model
#' `volume ~ b1 * (1 - exp(-b2 * pressure))`. Certified values are in
#' [nist_certified].
#' @format A data frame with 14 rows: `pressure`, `volume`.
#' @source NIST Statistical Reference Datasets (public domain),
#'   <https://www.itl.nist.gov/div898/strd/nls/nls_main.shtml>.
#' @examples
#' nl_fit(volume ~ pressure, misra1a, model = "exponential_rise")
"misra1a"

#' NIST BoxBOD: biochemical oxygen demand (higher difficulty)
#'
#' Six observations of oxygen demand over time (Box et al., 1978). Model
#' `bod ~ b1 * (1 - exp(-b2 * days))`. From NIST start 1 (b1 = 1, b2 = 1)
#' textbook Gauss-Newton diverges; see [nl_compare_solvers()].
#' @format A data frame with 6 rows: `days`, `bod`.
#' @source NIST Statistical Reference Datasets (public domain),
#'   <https://www.itl.nist.gov/div898/strd/nls/nls_main.shtml>.
#' @examples
#' nl_fit(bod ~ days, boxbod, model = "exponential_rise", start = c(a = 100, k = 0.75))
"boxbod"

#' NIST DanWood: radiated energy (lower difficulty)
#'
#' Energy radiated from a carbon filament lamp (Daniel and Wood, 1980). Model
#' `energy ~ b1 * temperature^b2`.
#' @format A data frame with 6 rows: `temperature` (1000 K), `energy`.
#' @source NIST Statistical Reference Datasets (public domain),
#'   <https://www.itl.nist.gov/div898/strd/nls/nls_main.shtml>.
#' @examples
#' nl_fit(energy ~ temperature, danwood, model = "power_law")
"danwood"

#' NIST Rat42: pasture yield (higher difficulty)
#'
#' Pasture yield against growing time (Ratkowsky, 1983). Logistic model
#' `yield ~ b1 / (1 + exp(b2 - b3 * time))`.
#' @format A data frame with 9 rows: `time`, `yield`.
#' @source NIST Statistical Reference Datasets (public domain),
#'   <https://www.itl.nist.gov/div898/strd/nls/nls_main.shtml>.
#' @examples
#' nl_fit(yield ~ time, rat42, model = "logistic")
"rat42"

#' NIST Rat43: onion bulb dry weight (higher difficulty)
#'
#' Dry weight of onion bulbs against growing time (Ratkowsky, 1983). Richards
#' model `weight ~ b1 / (1 + exp(b2 - b3 * time))^(1 / b4)`.
#' @format A data frame with 15 rows: `time`, `weight`.
#' @source NIST Statistical Reference Datasets (public domain),
#'   <https://www.itl.nist.gov/div898/strd/nls/nls_main.shtml>.
#' @examples
#' nl_fit(weight ~ time, rat43, model = "richards", start = c(b1 = 700, b2 = 5, b3 = 0.75, b4 = 1.3))
"rat43"

#' NIST MGH09: Kowalik and Osborne (higher difficulty)
#'
#' A rational enzyme-kinetics model with local minima (More, Garbow and
#' Hillstrom, 1981): `y ~ b1 * (x^2 + x * b2) / (x^2 + x * b3 + b4)`.
#' @format A data frame with 11 rows: `x`, `y`.
#' @source NIST Statistical Reference Datasets (public domain),
#'   <https://www.itl.nist.gov/div898/strd/nls/nls_main.shtml>.
#' @examples
#' nl_fit(y ~ b1 * (x^2 + x * b2) / (x^2 + x * b3 + b4), mgh09,
#'        start = c(b1 = 0.25, b2 = 0.39, b3 = 0.415, b4 = 0.39))
"mgh09"

#' Puromycin-treated enzyme reaction velocities
#'
#' The treated-cell half of R's [datasets::Puromycin] (Treloar, 1974), as used
#' by Bates and Watts (1988). Michaelis-Menten kinetics.
#' @format A data frame with 12 rows: `conc` (substrate, ppm), `rate`
#'   (counts/min^2).
#' @source Treloar, M. A. (1974), via Bates, D. M. and Watts, D. G. (1988).
#'   *Nonlinear Regression Analysis and Its Applications*. Wiley.
#' @examples
#' summary(nl_fit(rate ~ conc, puromycin, model = "michaelis_menten"))
"puromycin"

#' Simulated wheat yield response to nitrogen
#'
#' SIMULATED data: 7 nitrogen rates x 3 replicates, generated from the
#' Mitscherlich curve `yield = 9.5 - 6 * exp(-0.012 * nitrogen)` with normal
#' noise (sd 0.35).
#' @format A data frame with 21 rows: `nitrogen` (kg N/ha), `yield` (t/ha).
#' @source Simulated by the nlsfit Python package (seed 11).
#' @examples
#' nl_workflow(crop_yield_nitrogen, "nitrogen", "yield", models = "asymptotic_regression",
#'             plot = FALSE)
"crop_yield_nitrogen"

#' Simulated maize biomass growth
#'
#' SIMULATED data generated from a Gompertz curve
#' (`Asym = 180, b2 = 6, b3 = 0.92`, noise sd 4), so a logistic fit should
#' lose on AIC.
#' @format A data frame with 15 rows: `day` (days after sowing), `biomass`
#'   (g/plant).
#' @source Simulated by the nlsfit Python package (seed 23).
#' @examples
#' nl_workflow(plant_growth, "day", "biomass", plot = FALSE)$comparison
"plant_growth"

#' Simulated dose-response (cell viability)
#'
#' SIMULATED duplicate wells at 9 concentrations, generated from a
#' four-parameter logistic curve (`bottom = 5, top = 98, EC50 = 1.5,
#' hill = 1.2`, noise sd 3).
#' @format A data frame with 18 rows: `conc` (micromolar), `viability` (%).
#' @source Simulated by the nlsfit Python package (seed 5).
#' @examples
#' nl_fit(viability ~ conc, dose_response, model = "four_pl")
"dose_response"

#' Simulated drug elimination after an IV bolus
#'
#' SIMULATED plasma concentrations from first-order elimination
#' (`a = 12, k = 0.35`) with about 6% multiplicative noise: a good case for
#' weighted least squares.
#' @format A data frame with 10 rows: `time` (h), `conc` (mg/L).
#' @source Simulated by the nlsfit Python package (seed 7).
#' @examples
#' f <- nl_fit(conc ~ time, drug_elimination, model = "exponential_decay")
#' nl_fit(conc ~ time, drug_elimination, model = "exponential_decay",
#'        weights = 1 / fitted(f)^2)
"drug_elimination"

#' NIST certified values and starting points
#'
#' Certified parameter values, the two official starting points and the
#' certified residual sum of squares for the six bundled NIST problems.
#' @format A data frame with one row per parameter: `dataset`, `formula`,
#'   `parameter`, `certified`, `start1`, `start2`, `certified_sse`.
#' @source NIST Statistical Reference Datasets (public domain),
#'   <https://www.itl.nist.gov/div898/strd/nls/nls_main.shtml>.
#' @examples
#' subset(nist_certified, dataset == "boxbod")
"nist_certified"

#' Index of the bundled datasets
#'
#' @return A data frame with each dataset's name, kind (NIST, real or
#'   simulated), number of rows, x and y columns, the model that generated or
#'   is certified for the data, and a ready-to-run workflow call.
#' @examples
#' nl_datasets()
#' @export
nl_datasets <- function() {
  d <- data.frame(
    dataset = c("misra1a", "boxbod", "danwood", "rat42", "rat43", "mgh09", "puromycin",
                "crop_yield_nitrogen", "plant_growth", "dose_response", "drug_elimination"),
    kind = c(rep("NIST", 6), "real", rep("simulated", 4)),
    x = c("pressure", "days", "temperature", "time", "time", "x", "conc", "nitrogen", "day",
          "conc", "time"),
    y = c("volume", "bod", "energy", "yield", "weight", "y", "rate", "yield", "biomass",
          "viability", "conc"),
    generating_model = c("exponential_rise", "exponential_rise", "power_law", "logistic",
                        "richards", "(custom rational, see ?mgh09)", "michaelis_menten",
                        "asymptotic_regression", "gompertz", "four_pl", "exponential_decay"),
    stringsAsFactors = FALSE
  )
  env <- new.env()
  utils::data(list = d$dataset, package = "nlestim", envir = env)
  d$n <- vapply(d$dataset, function(nm) nrow(get(nm, envir = env)), 1L)
  d$workflow <- sprintf('nl_workflow(%s, "%s", "%s")', d$dataset, d$x, d$y)
  d[c("dataset", "kind", "n", "x", "y", "generating_model", "workflow")]
}
