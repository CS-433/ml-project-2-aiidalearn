using DrWatson
@quickactivate
using DFControl

function process_jobs(server, root)
    s = Server(server)
    for sys in filter(!isequal("data.json"), readdir(s, joinpath(s, root)))
        datfile = joinpath(s, root,sys, "data.json")
        locfile = datadir(sys, "data.json")
        if ispath(s, datfile)
            ispath(locfile) && rm(locfile)
            Servers.pull(s, datfile, locfile)
        end
    end
end

using ArgParse

function parse_cmdline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--server"
            help = "Server on which things are running."
            default = "fidis"
            required = false
            arg_type = String
        "--root","-r"
            help = "Root dir for runs on server."
            default = ""
            required = false
            arg_type = String
    end
    return parse_args(s)
end

function main()
    parsed_args = parse_cmdline()
    process_jobs(parsed_args["server"], parsed_args["root"])
end
main()
