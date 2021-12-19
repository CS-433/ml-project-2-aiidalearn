using DrWatson
@quickactivate
using HTTP
using JSON3
using DFControl

function pull_generate_jobs(nelements, nsites, api_key, server, root, args...)
    d = Dict("criteria"   => Dict("nelements" => nelements, "nsites" => nsites),
	                  "properties" => ["formula", "cif", "pretty_formula"])
    data = join(["$k=$(HTTP.escapeuri(JSON3.write(v)))" for (k, v) in d], "&")
    
    resp = HTTP.post("https://www.materialsproject.org/rest/v2/query",
		                          ["Content-Type" => "application/x-www-form-urlencoded", "x-api-key" => api_key],
					                       data)

    if resp.status != 200
        error("Something went wrong with your request: $(resp.status)")
    end

    valid_atsyms = keys(DFControl.Client.list_pseudoset(server, "sssp_efficiency"))
    
    for sys in filter(x -> all(y->Symbol(y) ∈ valid_atsyms, keys(x["formula"])),  unique(x -> x["formula"], JSON3.read(resp.body, Dict)["response"]))[601:end]
        sysname_old = sys["pretty_formula"]
        arr = [i > 1 && uppercase(c) == c ? "1"*c  : c  for (i, c) in enumerate(sysname_old)]
        arr_str = join(arr)
        if length(arr_str) == length(sysname_old)+1
            sysname = string(arr_str, "1")
        else 
            sysname = sysname_old
        end
        sysdir  = datadir(sysname)
        @info "Creating run for $sysname."
        mkpath(sysdir)
        cifpath = joinpath(sysdir, "$sysname.cif") 
        datapath = joinpath(sysdir, "data.json")
        write(cifpath, sys["cif"]) #Just save the cif file
        # run the jobs
        generate_jobs(cifpath, datapath, server, root, args...)
        
    end
end
            
function generate_jobs(cif_file, data_file, server, root, ecutwfcs, ecutrhos, kpoints, smearing, environment)
    name = splitext(splitpath(cif_file)[end])[1]
    dir = datadir(name)
    str = Structures.cif2structure(cif_file)

    #open data.json file and load in memory all the json dict
    data_string = read(data_file, String)
    datadict = JSON3.read(data_string, Vector{Dict})
    sums = map(x -> 10 * x["ecutwfc"] + 100 * x["ecutrho"] + 1000 * round(Int, 1/x["k_density"]), datadict)

    calc_template = Calculation{QE}(name = "scf",
                                    exec = Exec(exec    = "pw.x",
                                                dir     = "/work/theos/THEOS_software/QuantumESPRESSO/q-e-qe-6.7.0/bin",
                                                modules = ["intel", "intel-mpi", "intel-mkl"]))
    calc_template[:calculation] = "scf"
    calc_template[:conv_thr]    = 1e-9
    calc_template[:mixing_beta] = 0.4
    calc_template[:disk_io]     = "nowf"
    calc_template[:occupations] = "smearing"
    calc_template[:smearing]    = "mv"
    calc_template[:degauss]     = smearing
    
    job = Job(name, str, Calculation[], server=server, environment = environment, dir = joinpath(root, name))
    set_pseudos!(job, :sssp_efficiency)
    for ecutwfc in ecutwfcs
        for ecutrho in ecutrhos
            if ecutrho == ecutwfc
                continue
            end
            for nk in kpoints
                calc = deepcopy(calc_template)
                set_name!(calc, "scf_$(ecutwfc)_$(ecutrho)_$(nk)", print=false)
                Calculations.set_flags!(calc, :ecutrho => ecutrho, :ecutwfc => ecutwfc, print=false)
                set_kpoints!(calc, (nk, nk, nk), print=false)
                push!(job, calc)
                #check if current ecutwfc, ecutrho and nk is present
                sum = 10 * ecutwfc + 100 * ecutrho + 1000 * nk
                job["scf_$(ecutwfc)_$(ecutrho)_$(nk)"].run = !(sum in sums)

            end
        end
    end
    submit(job)
end


using ArgParse

ArgParse.parse_item(::Type{StepRange}, x::AbstractString) = StepRange(parse.(Int, split(x, ":"))...)

function parse_cmdline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--nelements", "-e"
            help = "number of elements"
            arg_type = Int
            required = true
        "--nsites", "-s"
            help = "number of sites per unit cell"
            arg_type = Int
            required = true
        "--apikey", "-a"
            help = "Materialsproject API key"
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
        "--smearing"
            help = "Smearing strength"
            default = 0.02
            required=false
            arg_type=Float64
        "--server"
            help = "Server to run on."
            default = "fidis"
            required = false
            arg_type = String
        "--root"
            help = "Root dir for runs on server."
            default = ""
            required = false
            arg_type = String
        "--environment"
            help = "environment to use."
            default = "normal_1node"
            required = false
            arg_type = String
    end
    return parse_args(s)
end

function main()
    parsed_args = parse_cmdline()
    pull_generate_jobs(parsed_args["nelements"], parsed_args["nsites"], parsed_args["apikey"], parsed_args["server"], parsed_args["root"], parsed_args["ecutwfc"], parsed_args["ecutrho"], parsed_args["kpoints"], parsed_args["smearing"], parsed_args["environment"])
end

main()

