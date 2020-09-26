# *******************************************
#  Checker
# *******************************************
function check_config_consistency(c1::ConfigFFT, c2::ConfigFFT)
    if !(c1 === c2)
        errmsg = "ConfigFFTInconsistencyError"
        throw(ErrorException(errmsg))
    end
end


function check_size_consistency(
        vals::Array, config::ConfigFFT)

    if size(vals) != config.ngrids
        errmsg = (
            "size(vals) and config.ngrids must match:\n"
            * "size(vals)=$(size(vals)), and "
            * "config.ngrids=$(config.ngrids)"
        )
        throw(ErrorException(errmsg))
    end
end
