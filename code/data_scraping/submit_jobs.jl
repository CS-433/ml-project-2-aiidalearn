using DrWatson
@quickactivate
using DFControl
using ArgParse

function generate_jobs(cif_file, ecutwfcs, ecutrhos, kpoints, smearing)
    name = splitext(splitpath(cif_file)[end])[1]
    dir = datadir(name)
    mkpath(dir)
    cp(cif_file, joinpath(dir, name*".cif"))
    str = Structures.cif2structure(cif_file)

    calc = Calculation[Calculation{QE}(name="scf", exec=Exec(exec="pw.x", dir="/work/theos/THEOS_software/QuantumESPRESSO/q-e-qe-6.7.0/bin", modules=["intel", "intel-mpi", "intel-mkl"]))]
    Calculations.set_flags!(calc[1].exec, :nk => 10)
    
    job = Job(name, str, calc, server="fidis", environment ="normal_10nodes")
    set_pseudos!(job, :sssp_efficiency)
    for ecutwfc in ecutwfcs
        for ecutrho in ecutrhos
            if ecutrho == ecutwfc
                continue
            end
            for nk in kpoints

                tj = deepcopy(job)
                tj.dir = "$name/$ecutwfc/$ecutrho/$nk"
                tj[:ecutrho] = ecutrho
                tj[:ecutwfc] = ecutwfc
                tj[:occupations] = "smearing"
                tj[:smearing] = "mv"
                tj[:degauss] = smearing
                set_kpoints!(tj["scf"], (nk, nk, nk))
                submit(tj)
            end
        end
    end
end

ArgParse.parse_item(::Type{StepRange}, x::AbstractString) = StepRange(parse.(Int, split(x, ":"))...)

function parse_cmdline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--cif", "-c"
            help = "Cif file with the structure"
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
        "--smearing", "-s"
            help = "Smearing strength"
            default = 0.02
            required=false
            arg_type=Float64
    end
    return parse_args(s)
end

function main()
    parsed_args = parse_cmdline()
    generate_jobs(parsed_args["cif"], parsed_args["ecutwfc"], parsed_args["ecutrho"], parsed_args["kpoints"], parsed_args["smearing"])
end
main()
