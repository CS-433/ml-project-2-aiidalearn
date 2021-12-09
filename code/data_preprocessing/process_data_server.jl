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
        d1 == "data.json" && continue
        dir1 = joinpath(sys, d1)
        ecutwfc = parse(Int, d1)
        for d2 in readdir(dir1)
            dir2 = joinpath(dir1, d2)
            ecutrho = parse(Int, d2)
            for d3 in readdir(dir2)
                dir3 = joinpath(dir2, d3)
                nk = parse(Int, d3)
                scf_file = joinpath(dir3, "scf.out")
                if ispath(scf_file) && mtime(scf_file) > age 
                    outdata = DFC.FileIO.qe_read_pw_output(joinpath(dir3, "scf.out"))
                    if haskey(outdata, :accuracy) && haskey(outdata, :fermi) && haskey(outdata, :timing)
                        outd = Dict("k_density" => 1/nk, "ecutrho" => ecutrho, "ecutwfc" => ecutwfc)
                        outd["accuracy"]=outdata[:accuracy][end]
                        outd["total_energy"]=outdata[:total_energy][end]/2
                        outd["n_iterations"] = outdata[:scf_iteration][end]
                        outd["converged"] = outdata[:converged]
                        outd["fermi"] = outdata[:fermi]
                        outd["time"] = Dates.toms(outdata[:timing][1].wall + outdata[:timing][2].wall)
                        push!(results, outd)
                    end
                end
            end
        end
    end
    @info "$(nres - length(results)) new results"
    !isempty(results) && JSON3.write(datfile, results)
end
