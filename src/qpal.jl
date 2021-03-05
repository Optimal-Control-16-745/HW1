using StaticArrays
using LinearAlgebra

mutable struct QPAL{n,m,p,nn,mn,pn,T}
    P::Symmetric{T,SMatrix{n,n,T,nn}}
    q::SVector{n,T}
    A::SMatrix{m,n,T,mn}
    b::SVector{m,T}
    C::SMatrix{p,n,T,pn}
    d::SVector{p,T}
    a::MVector{p,Bool}  # active set
    ρ::T   # penalty value
    ϕ::T   # penalty scaling
end

function QPAL(qp::QPData, ρ, ϕ)
    n,m,p = size(qp)
    P = Symmetric(SMatrix{n,n}(qp.P))
    q = SVector{n}(qp.q)
    A = SMatrix{m,n}(qp.A)
    b = SVector{m}(qp.b)
    C = SMatrix{p,n}(qp.C)
    d = SVector{p}(qp.d)
    a = @MVector zeros(Bool,p)
    QPAL(P,q,A,b,C,d,a,ρ,ϕ)
end

objective(qp::QPAL, x) = 0.5 * x'qp.P*x + qp.q'x
ceq(qp::QPAL, x) = qp.A * x - qp.b
cin(qp::QPAL, x) = qp.C * x - qp.d

function primal_residual(qp::QPAL{n,m,p}, x, λ, μ) where {n,m,p}
    r = qp.P*x + qp.q
    m > 0 && (r += qp.A'λ)
    p > 0 && (r += qp.C'μ)
    return r
end

function dual_residual(qp::QPAL, x, λ, μ)
    g = ceq(qp, x)
    h = cin(qp, x)
    return [g; Πpos(h)]
end

function auglag(solver::QPAL, x, λ, μ)
    ρ = solver.ρ
    J = objective(solver, x)
    g = ceq(solver, x)
    h = cin(solver, x)
    active_set!(solver, h, μ)
    Iρ = ρ*Diagonal(SVector(solver.a))
    J += λ'g + μ'h + 0.5*ρ*g'g + 0.5*h'Iρ*h
end

function algrad(solver::QPAL{n,m,p}, x, λ, μ) where {n,m,p}
    P,A,C = solver.P, solver.A, solver.C
    ρ = solver.ρ
    grad = P*x + solver.q
    if m > 0
        g = ceq(solver, x)
        grad += A'λ + ρ*A'g
    end
    if p > 0
        h = cin(solver, x)
        active_set!(solver, h, μ)
        Iρ = ρ*Diagonal(SVector(solver.a))
        grad += C'μ + C'*(Iρ*h)
    end
end

function alhess(solver::QPAL{n,m,p}, x, λ, μ) where {n,m,p}
    P,A,C = solver.P, solver.A, solver.C
    ρ = solver.ρ
    H = P
    if m > 0
        H += ρ*(A'A)
    end
    if p > 0
        Iρ = Diagonal(SVector(solver.a))
        H += ρ*(C'Iρ*C)
    end
    return H
end

function active_set!(solver::QPAL{n,m,p}, h, μ) where {n,m,p}
    for i = 1:p
        solver.a[i] = μ[i] > 0 || h[i] >= 0
    end
end

function newton_step(solver::QPAL, x, λ, μ)
    r = algrad(solver, x, λ, μ)
    H = alhess(solver, x, λ, μ)
    cholH = cholesky(H)
    -(cholH\r)
end

function dual_update(solver::QPAL, x, λ, μ)
    ρ = solver.ρ
    λnext = λ + ρ*ceq(solver, x)
    μnext = Πpos(μ + ρ*cin(solver, x))
    return λnext, μnext
end

@generated function Πpos(x::StaticVector{n,T}) where {n,T}
    expr = [:(max(0,x[$i])) for i = 1:n]
    return :(SVector{n}($(expr...)))
end

function inner_solve(solver::QPAL, x, λ, μ; ϵ=1e-6, max_iters=10, verbose=1)
    for i = 1:max_iters
        r = algrad(solver, x, λ, μ)
        verbose > 1 && println(norm(r))
        if norm(r) < ϵ
            return x
        end
        H = alhess(solver, x, λ, μ)
        dx = -H\r
        # TODO: line search
        x += dx 
    end
    @warn "Inner solve max iterations"
    return x
end

function solve(solver::QPAL, x, λ, μ; 
        max_iters=10, 
        eps_primal=1e-6, 
        eps_inner=1e-6, 
        verbose=0,
        ρ0 = solver.ρ
    )
    solver.ρ = ρ0
    for i = 1:max_iters
        x = inner_solve(solver, x, λ, μ, ϵ=eps_inner, verbose=verbose)
        λ, μ = dual_update(solver, x, λ, μ)
        solver.ρ *= solver.ϕ   # penalty update
        v = dual_residual(solver, x, λ, μ)
        verbose > 0 && println("Outer Iter $i: feas = ", norm(v))
        if norm(v) < eps_primal
            return x, λ, μ
        end
    end
    @warn "Outer loop max iterations"
    return x, λ, μ
end