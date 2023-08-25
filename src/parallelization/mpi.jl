using MPI

struct MPIParallelizer <: CounterfactualExplanations.AbstractParallelizer end

"""
    split_count(N::Integer, n::Integer)

Return a vector of `n` integers which are approximately equally sized and sum to `N`. Lifted from https://juliaparallel.org/MPI.jl/v0.20/examples/06-scatterv/.
"""
function split_count(N::Integer, n::Integer)
    q, r = divrem(N, n)
    return [i <= r ? q + 1 : q for i in 1:n]
end

"""
    split_obs(obs::AbstractVector, n::Integer)

Return a vector of `n` group indices for `obs`.
"""
function split_obs(obs::AbstractVector, n::Integer)
    N = length(obs)
    N_counts = split_count(N, n)
    _start = cumsum([1; N_counts[1:(end - 1)]])
    _stop = cumsum(N_counts)
    return [obs[_start[i]:_stop[i]] for i in 1:n]
end

"""
    parallelize(
        parallelizer::MPIParallelizer,
        f::Function,
        args...;
        kwargs...,
    )

A function that can be used to multi-process the evaluation of `f`. The function `f` should be a function that takes a single argument. The argument should be a vector of counterfactual explanations. The function will split the vector of counterfactual explanations into groups of approximately equal size and distribute them to the processes. The results are then collected and returned.
"""
function CounterfactualExplanations.parallelize(
    parallelizer::MPIParallelizer,
    f::Function,
    args...;
    kwargs...,
)

    @assert CounterfactualExplanations.parallelizable(f) "`f` is not a parallelizable process."

    # Setup:
    collection = args[1] |> x -> vectorize_collection(x)
    if length(args) > 1
        _args = args[2:end]
    end

    # MPI:
    @info "Using `MPI.jl` for multi-processing."
    MPI.Init()

    comm = MPI.COMM_WORLD                               # Collection of processes that can communicate in our world 🌍
    rank = MPI.Comm_rank(comm)                          # Rank of this process in the world 🌍
    n_proc = MPI.Comm_size(comm)                        # Number of processes in the world 🌍

    chunks = split_obs(collection, n_proc)                     # Split ces into groups of approximately equal size
    item = MPI.scatter(chunks, comm)                      # Scatter ces to all processes
    if length(args) > 1
        output = f(item, _args...; kwargs...)                           # Evaluate ces on each process
    else
        output = f(item; kwargs...)                           # Evaluate ces on each process
    end

    MPI.Barrier(comm)                                   # Wait for all processes to reach this point

    # Collect output from all processes:
    if rank == 0
        output = MPI.gather(output, comm)
        output = vcat(output...)
    end

    return output
end