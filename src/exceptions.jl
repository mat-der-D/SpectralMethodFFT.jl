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

function check_length_consistency(
        array1::Array, array2::Array
    )

    if length(array1) != length(array2)
        errmsg = (
            "lengths of arrays must be equal"
        )
        throw(ErrorException(errmsg))
    end

end

"""
warn_if_ngrid_is_not_power_of_2 displays warning
if each ngrid in ngrids is not power of 2,
for FFTW efficiency.
"""
function warn_if_ngrid_is_not_power_of_2(ngrids)
    for (axis, ngrid) in enumerate(ngrids)
        if !is_power_of_2(ngrid)
            println(
                "WARNING:",
                " ngrids[", axis, "]=", ngrid,
                " should be power of 2 for FFTW efficiency."
            )
        end
    end
end


function is_power_of_2(num::Int)

    if num<=0
        errmsg="num must be >0"
        throw(DomainError(num, errmsg))
    elseif num%2!=0
        return num==1
    else
        return is_power_of_2(num÷2)
    end

end
