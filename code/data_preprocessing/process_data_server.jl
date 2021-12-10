using JSON3
using DFControl
using Dates

for sys in filter(isdir, readdir()) # Structure
    @info "Processing $sys."
    datfile = joinpath(sys, "data.json")
    if ispath(datfile)
        age = mtime(datfile) 
        results = JSON3.read(read(datfile, String), Vector{Dict})
    else
        age = 0
        results = []
    end
    nres = length(results)
    for d1 in readdir(sys)
        d1 == "data.json" || isdir(d1) && continue
        if occursin("scf", d1) && occursin(".out", d1)
            file = joinpath(sys, d1)
            if mtime(file) > age
                filename = splitext(d1)[1]
                ecutwfc, ecutrho, nk = parse.(Int, split(filename, "_")[2:end])
                outdata = DFC.FileIO.qe_read_pw_output(file)
                if haskey(outdata, :accuracy) && haskey(outdata, :fermi) && haskey(outdata, :timing)
                    outd = Dict("k_density" => 1/nk, "ecutrho" => ecutrho, "ecutwfc" => ecutwfc)
                    outd["accuracy"]     = outdata[:accuracy][end]
                    outd["total_energy"] = outdata[:total_energy][end]/2
                    outd["n_iterations"] = outdata[:scf_iteration][end]
                    outd["converged"]    = outdata[:converged]
                    outd["fermi"]        = outdata[:fermi]
                    outd["time"]         = Dates.toms(outdata[:timing][1].wall + outdata[:timing][2].wall)
                    push!(results, outd)
                end
            end
        end
    end
    @info "$(length(results) - nres) new results."
    !isempty(results) && JSON3.write(datfile, results)
end
