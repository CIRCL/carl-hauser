ROUND_DECIMAL = 5
END_LINE = ", "

def print_stats_human(stats_result, round_decimal = ROUND_DECIMAL):
    tmp_str = ""
    tmp_str += "nobs : " + str( getattr(stats_result, "nobs")) + " " + END_LINE
    tmp_str += "min time : " + str(round(getattr(stats_result, "minmax")[0],round_decimal)) + "s " + END_LINE
    tmp_str += "max time : " + str(round(getattr(stats_result, "minmax")[1],round_decimal)) + "s " + END_LINE
    tmp_str += "mean :" + str(round(getattr(stats_result, "mean"),round_decimal)) + "s " + END_LINE
    tmp_str += "variance : " + str(round(getattr(stats_result, "variance"),round_decimal)) + "s " + END_LINE
    tmp_str += "skewness : " + str(round(getattr(stats_result, "skewness"),round_decimal) ) + "s " + END_LINE
    tmp_str += "kurtosis : " + str(round(getattr(stats_result, "kurtosis") ,round_decimal))
    print(tmp_str)

def print_stats(stats_result, round_decimal = ROUND_DECIMAL):
    tmp_str = ""
    tmp_str += "Nobs & Min time & Max time & Mean & Variance & Skewness & Kurtosis & Quality\\ \hline \n"
    tmp_str += str( getattr(stats_result, "nobs")) + " & "
    tmp_str += str(round(getattr(stats_result, "minmax")[0],round_decimal)) + " & "
    tmp_str += str(round(getattr(stats_result, "minmax")[1],round_decimal)) + " & "
    tmp_str += str(round(getattr(stats_result, "mean"),round_decimal)) + " & "
    tmp_str += str(round(getattr(stats_result, "variance"),round_decimal)) + " & "
    tmp_str += str(round(getattr(stats_result, "skewness"),round_decimal) ) + " & "
    tmp_str += str(round(getattr(stats_result, "kurtosis") ,round_decimal))
    tmp_str += str("\\\ \hline \n")
    print(tmp_str)