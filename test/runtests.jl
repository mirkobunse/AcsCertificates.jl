using LossFunctions, Random, Test
import AcsCertificates.Certificates, AcsCertificates.Data

@testset "Data.subsample_indices" begin
    for _ in 1:100
        desired_pY = rand() * .45 + .05
        current_pY = rand() * .45 + .05
        m = 250 + round(Int, rand() * 1000)
        y = shuffle(vcat(
            ones(Int, round(Int, current_pY * m)),
            -1 * ones(Int, round(Int, (1-current_pY) * m))
        ))
        i_s = Data.subsample_indices(y, desired_pY)
        y_s = y[i_s]
        @test all(abs.(y_s) .== 1)
        @test length(y_s) <= length(y)
        @test sum(y_s.==1)/length(y_s) ≈ desired_pY  atol=0.01
    end
end

"""
    TestCase(L, m, p, ℓ_1, ℓ_2, w_1=1, w_2=1) -> t

Generate random ground-truth labels `t.y` and predictions
`t.y_h` for the total number `m` of samples, class
proportions `p`, and class-wise losses `ℓ_1` and `ℓ_2`
under the loss function `L` weighted by `(w_1, w_2)`.
"""
struct TestCase{T <: Real}
    y_h::Vector{T}
    y::Vector{Int}
    w_y::Vector{Float64}
end

function TestCase(
        L::ZeroOneLoss,
        m::Int,
        p::Real,
        ℓ_1::Real,
        ℓ_2::Real,
        w_1::Real=1.0,
        w_2::Real=1.0)
    m_1 = round(Int, m*(1-p)) # negative class
    m_2 = round(Int, m*p) # positive class
    y = shuffle(vcat(-1*ones(Int, m_1), ones(Int, m_2)))
    y_h = copy(y) # derive predictions from ground-truth
    l_1 = round(Int, m_1*ℓ_1) # number of misclassifications
    l_2 = round(Int, m_2*ℓ_2)
    y_h[shuffle(collect(1:m)[y.==-1])[1:l_1]] .= 1
    y_h[shuffle(collect(1:m)[y.==1])[1:l_2]] .= -1
    return TestCase(y_h, y, [w_1, w_2])
end

TestCase(L::SupervisedLoss, args...) =
    error("random_data not yet implemented for $L")

L = ZeroOneLoss() # all tests use the ZeroOneLoss

@testset "Onesided Δℓ_Result typing" begin
    for t in [
            TestCase(L, 200, .1, .19, .2), # labels not switched
            TestCase(L, 200, .1, .2, .19) # labels SWITCHED
            ]
        c = Certificates.Certificate(L, t.y_h, t.y; w_y=t.w_y)
        Base.show(stdout, "text/plain", c)
        @test any(isnan.(c.Δℓ.ϵ_y)) # one of the ϵ values should be NaN (one-sided result)
        @test any(isfinite.(c.Δℓ.ϵ_y)) # and one should be a finite real number
        @test typeof(c.Δℓ) == Certificates.Δℓ_Result{Certificates.OneSided_Δℓ_MinMax}
    end
end

# how to test: compute certificates and print them to the console
function certify(t::TestCase)
    println(stdout, "") # empty line before each test
    c = Certificates.Certificate(L, t.y_h, t.y; w_y=t.w_y)
    Base.show(stdout, "text/plain", c)
    acquire(Certificates.Certificate(L, t.y_h, t.y; w_y=t.w_y, allow_onesided=false))
end

N = 100
PRIORS = [
    (.2, .8),
    (2., 8.),
    (8., 2.),
    (.5, .5),
    (5., 5.)
]

acquire(c::Certificates.Certificate) = for (α, β) in PRIORS
    println(stdout,
        "α=$α, β=$β => ",
        Certificates.suggest_acquisition(c, N, α, β))
end

# conduct tests
@testset "LARGE sample, MODERATE loss difference" for t in [
        TestCase(L, 10_000, .5, .1, .2),
        TestCase(L, 10_000, .1, .1, .2),
        TestCase(L, 10_000, .9, .1, .2)]
    certify(t)
end
@testset "LARGE sample, MODERATE loss difference (LABELS SWITCHED)" for t in [
        TestCase(L, 10_000, .5, .2, .1),
        TestCase(L, 10_000, .9, .2, .1),
        TestCase(L, 10_000, .1, .2, .1)]
    certify(t)
end
@testset "LARGE sample, MODERATE loss difference (WEIGHTED)" for t in [
        TestCase(L, 10_000, .5, .1, .2, 1., .52),
        TestCase(L, 10_000, .1, .1, .2, 1., .52),
        TestCase(L, 10_000, .9, .1, .2, 1., .52)]
    certify(t)
end
@testset "LARGE sample, TINY loss difference" for t in [
        TestCase(L, 10_000, .5, .19, .2),
        TestCase(L, 10_000, .1, .19, .2),
        TestCase(L, 10_000, .9, .19, .2)]
    certify(t)
end
@testset "HUGE sample, TINY loss difference" for t in [
        TestCase(L, 1_000_000, .5, .19, .2),
        TestCase(L, 1_000_000, .1, .19, .2),
        TestCase(L, 1_000_000, .9, .19, .2)]
    certify(t)
end
@testset "SMALL sample, MODERATE loss difference" for t in [
        TestCase(L, 2_000, .5, .1, .2),
        TestCase(L, 2_000, .1, .1, .2),
        TestCase(L, 2_000, .9, .1, .2)]
    certify(t)
end
@testset "LARGE sample, HIGH loss difference" for t in [
        TestCase(L, 10_000, .5, .05, .2),
        TestCase(L, 10_000, .1, .05, .2),
        TestCase(L, 10_000, .9, .05, .2)]
    certify(t)
end

# the following certificates fail due to sample size
@testset "TINY sample, HIGH loss difference" for t in [
        TestCase(L, 200, .5, .05, .2),
        TestCase(L, 200, .1, .05, .2),
        TestCase(L, 200, .9, .05, .2)]
    certify(t)
end
@testset "TINY sample, MODERATE loss difference" for t in [
        TestCase(L, 200, .5, .1, .2),
        TestCase(L, 200, .1, .1, .2),
        TestCase(L, 200, .9, .1, .2)]
    certify(t)
end
@testset "TINY sample, MODERATE loss difference (LABELS SWITCHED)" for t in [
        TestCase(L, 200, .5, .2, .1),
        TestCase(L, 200, .9, .2, .1),
        TestCase(L, 200, .1, .2, .1)]
    certify(t)
end
