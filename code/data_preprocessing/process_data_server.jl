using JSON3
using DFControl
using Dates

dirs = filter(x->ispath(joinpath(x,"scf.out")), getindex.(registered_jobs("", "localhost"),1))
systems = unique(map(x-> splitpath(x)[4], dirs))
Threads.@threads for sys in systems # Structure
    @info "Processing $sys."
    dirs = filter(x->ispath(joinpath(x,"scf.out")), getindex.(registered_jobs(sys, "localhost"),1))
    
    datfile = joinpath(sys, "data.json")
    if ispath(datfile)
        age = mtime(datfile) 
        results = JSON3.read(read(datfile, String), Vector{Dict})
    else
        age = 0
        results = []
    end
    for d in dirs
        if mtime(joinpath(d, "scf.out")) > age
            outdata = DFC.FileIO.qe_read_pw_output(joinpath(d, "scf.out"))
            if haskey(outdata, :accuracy) && haskey(outdata, :fermi) && haskey(outdata, :timing)
                ecutwfc, ecutrho, nk = parse.(Int, splitpath(d)[end-2:end]) 
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
    @show length(results)
    !isempty(results) && JSON3.write(datfile, results)
end
