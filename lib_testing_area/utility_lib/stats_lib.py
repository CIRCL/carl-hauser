ROUND_DECIMAL = 5
END_LINE = ", "
import configuration
import results
import logging
import pathlib
import pprint
from scipy import stats

class Stats_handler():
    def __init__(self, conf: configuration.Default_configuration):
        self.conf = conf
        self.logger = logging.getLogger(__name__)

    def print_stats_human(self, conf: configuration.Default_configuration, results : results.RESULTS, round_decimal=ROUND_DECIMAL):
        stats_result = stats.describe(results.TIME_LIST_MATCHING)

        tmp_str = ""
        tmp_str += "nobs : " + str(getattr(stats_result, "nobs")) + " " + END_LINE
        tmp_str += "min time : " + str(round(getattr(stats_result, "minmax")[0], round_decimal)) + "s " + END_LINE
        tmp_str += "max time : " + str(round(getattr(stats_result, "minmax")[1], round_decimal)) + "s " + END_LINE
        tmp_str += "mean :" + str(round(getattr(stats_result, "mean"), round_decimal)) + "s " + END_LINE
        tmp_str += "variance : " + str(round(getattr(stats_result, "variance"), round_decimal)) + "s " + END_LINE
        tmp_str += "skewness : " + str(round(getattr(stats_result, "skewness"), round_decimal)) + "s " + END_LINE
        tmp_str += "kurtosis : " + str(round(getattr(stats_result, "kurtosis"), round_decimal))
        self.logger.info(tmp_str)

    def print_stats(self, conf: configuration.Default_configuration, results : results.RESULTS, round_decimal=ROUND_DECIMAL):
        stats_result = stats.describe(results.TIME_LIST_MATCHING)

        tmp_str = ""
        tmp_str += "Nobs & Min time & Max time & Mean & Variance & Skewness & Kurtosis & Quality\\ \hline \n"
        tmp_str += str(getattr(stats_result, "nobs")) + " & "
        tmp_str += str(round(getattr(stats_result, "minmax")[0], round_decimal)) + " & "
        tmp_str += str(round(getattr(stats_result, "minmax")[1], round_decimal)) + " & "
        tmp_str += str(round(getattr(stats_result, "mean"), round_decimal)) + " & "
        tmp_str += str(round(getattr(stats_result, "variance"), round_decimal)) + " & "
        tmp_str += str(round(getattr(stats_result, "skewness"), round_decimal)) + " & "
        tmp_str += str(round(getattr(stats_result, "kurtosis"), round_decimal))
        tmp_str += str("\\\ \hline \n")
        self.logger.info(tmp_str)

    def write_stats_to_folder(self, conf: configuration.Default_configuration, results : results.RESULTS):
        fn = "stats.txt"
        filepath = conf.OUTPUT_DIR / fn
        with filepath.open("w", encoding="utf-8") as f:
            f.write(pprint.pformat(vars(results)))

        self.logger.debug(f"Statistics file saved as {filepath}.")
