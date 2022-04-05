"""
"""
function print_vector(x::Vector{Float64}, msg::String = "")
    @printf("%s\n", msg)
    for (i,v) in enumerate(x)
        @printf("    %+.6f", v)
        if i % 5 == 0
            @printf("\n")
        end
    end
    @printf("\n")
end

"""
"""
function dropzeros!(x::Vector{Float64}, eps::Float64 = 1.0e-10)
    for (i,v) in enumerate(x)
        if abs(v) < eps
            x[i] = 0.0
        end
    end
end

function print_matrix(A::SparseMatrixCSC{Float64, Int64})
    for i = 1:A.m
        a = A[i,:]
        SparseArrays.droptol!(a, 1.0e-10)
        if length(a.nzind) > 0
            @printf("row%6d", i)
            for (k,j) in enumerate(a.nzind)
                @printf("\tcol%6d\t%+.6e", j, a.nzval[k])
            end
            @printf("\n")
        end
    end
end
