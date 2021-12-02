using DrWatson
@quickactivate
using DFControl

using Dates
using JSON3
function process_jobs(name, ecutwfcs, ecutrhos, kpoints)
    results = []
    s = Server("fidis")
    for ecutwfc in ecutwfcs
        for ecutrho in ecutrhos
            if ecutrho == ecutwfc
                continue
            end
            for nk in kpoints
                @info "Processing $name, $ecutwfc, $ecutrho, $nk"
                tj = Job("$name/$ecutwfc/$ecutrho/$nk", "fidis")
               
                if ispath(s, joinpath(tj, "CRASH"))
                    @warn "Job $(tj.dir) crashed, see CRASH file for more info."
                elseif !ispath(s, joinpath(tj, "scf.out"))
                    @warn "Something went wrong for job $(tj.dir), resubmitting."
                    submit(tj)
                else
                    outdata = outputdata(tj)
                    if haskey(outdata["scf"], :accuracy) && haskey(outdata["scf"], :fermi) && haskey(outdata["scf"], :timing)
                        outd = Dict("k_density" => 1/nk, "ecutrho" => ecutrho, "ecutwfc" => ecutwfc)
                        
                        outd["accuracy"]=outdata["scf"][:accuracy][end]
                        outd["total_energy"]=outdata["scf"][:total_energy][end]/length(tj.structure.atoms)
                        outd["n_iterations"] = outdata["scf"][:scf_iteration][end]
                        outd["converged"] = outdata["scf"][:converged]
                        outd["fermi"] = outdata["scf"][:fermi]
                        outd["time"] = Dates.toms(outdata["scf"][:timing][1].wall + outdata["scf"][:timing][2].wall)
                        push!(results, outd)
                    else
                        @warn "Something went wrong for job $(tj.dir) during scf run, resubmitting."
                        submit(tj)
                    end
                end
            end
        end
    end
    mkpath(datadir(name))
    @info "Saving data to $(datadir(name, "data.json"))."
    JSON3.write(datadir(name, "data.json"), results)
    return results
end

using ArgParse

ArgParse.parse_item(::Type{StepRange}, x::AbstractString) = StepRange(parse.(Int, split(x, ":"))...)

function parse_cmdline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--name", "-n"
            help = "Name of the job (e.g. NaCl)"
            arg_type = String
            required = true
        "--kpoints", "-k"
            help = "3 numbers designating kstart:kstep:kend"
            default = 2:2:10
            required = false
            arg_type = StepRange
        "--ecutwfc", "-w"
            help = "3 numbers designating ecutwfc_start:ecutwfc_step:ecutwfc_end"
            default = 20:5:100
            required = false
            arg_type = StepRange
        "--ecutrho", "-r"
            help = "3 numbers designating ecutrho_start:ecutrho_step:ecutrho_end"
            default = 100:40:400
            required = false
            arg_type = StepRange
    end
    return parse_args(s)
end

function main()
    parsed_args = parse_cmdline()
    process_jobs(parsed_args["name"], parsed_args["ecutwfc"], parsed_args["ecutrho"], parsed_args["kpoints"])
end
main()
