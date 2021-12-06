using DrWatson
@quickactivate
using HTTP
using JSON3
using DFControl

function pull_generate_jobs(nelements, nsites, api_key, args...)
    d = Dict("criteria"   => Dict("nelements" => nelements, "nsites" => nsites),
	                  "properties" => ["formula", "cif", "pretty_formula"])
    data = join(["$k=$(HTTP.escapeuri(JSON3.write(v)))" for (k, v) in d], "&")
    
    resp = HTTP.post("https://www.materialsproject.org/rest/v2/query",
		                          ["Content-Type" => "application/x-www-form-urlencoded", "x-api-key" => api_key],
					                       data)

    if resp.status != 200
        error("Something went wrong with your request: $(resp.status)")
    end

    valid_atsyms = keys(DFControl.Client.list_pseudoset("fidis", "sssp_efficiency"))
    
    for sys in filter(x -> all(y->Symbol(y) ∈ valid_atsyms, keys(x["formula"])),  unique(x -> x["formula"], JSON3.read(resp.body, Dict)["response"]))[350:400]
        sysname = sys["pretty_formula"]
        sysdir  = datadir(sysname)
        @info "Creating run for $sysname."
        mkpath(sysdir)
        cifpath = joinpath(sysdir, "$sysname.cif") 
        write(cifpath, sys["cif"]) #Just save the cif file
        # run the jobs
        generate_jobs(cifpath, args...)
    end
end
            
function generate_jobs(cif_file, ecutwfcs, ecutrhos, kpoints, smearing)
    name = splitext(splitpath(cif_file)[end])[1]
    dir = datadir(name)
    str = Structures.cif2structure(cif_file)

    calc = Calculation[Calculation{QE}(name="scf", exec=Exec(exec="pw.x", dir="/work/theos/THEOS_software/QuantumESPRESSO/q-e-qe-6.7.0/bin", modules=["intel", "intel-mpi", "intel-mkl"]))]
    calc[1][:calculation] = "scf"
    calc[1][:conv_thr] = 1e-9
    calc[1][:mixing_beta] = 0.4
    calc[1][:disk_io] = "nowf"
#Calculations.set_flags!(calc[1].exec, :nk => 10)
    
    job = Job(name, str, calc, server="fidis", environment ="normal_1nodes")
    server = Server("fidis")
    set_pseudos!(job, :sssp_efficiency)
    jobs = Job[]
    for ecutwfc in ecutwfcs
        for ecutrho in ecutrhos
            if ecutrho == ecutwfc
                continue
            end
            for nk in kpoints
                dir = "ml_project/$name/$ecutwfc/$ecutrho/$nk"
                if !ispath(server, dir) || !ispath(server, joinpath(dir, "scf.out"))
                    tj = deepcopy(job)
                    tj.dir = dir 
                    tj[:ecutrho] = ecutrho
                    tj[:ecutwfc] = ecutwfc
                    tj[:occupations] = "smearing"
                    tj[:smearing] = "mv"
                    tj[:degauss] = smearing
                    set_kpoints!(tj["scf"], (nk, nk, nk))
                    push!(jobs, tj)
                end
            end
        end
    end
    submit(jobs)
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
    end
    return parse_args(s)
end

function main()
    parsed_args = parse_cmdline()
    pull_generate_jobs(parsed_args["nelements"], parsed_args["nsites"], parsed_args["apikey"], parsed_args["ecutwfc"], parsed_args["ecutrho"], parsed_args["kpoints"], parsed_args["smearing"])
end

main()

