using Revise, AcsCertificates
using Plots 
using Distributions
using DataFrames
using CSV

mean = 0.8
sd = 0.05 #standardabweichung
a,b = AcsCertificates.Certificates.beta_parameters(1-mean, sd)
d = Distributions.Beta(a, b)
X = collect(range(0.0, 1.0, length=500))
Y = map(x -> Distributions.pdf(d, x), X)
fig = plot(X,Y)
savefig(fig, "A_low_pdf")
Y = map(x -> Distributions.cdf(d, x), X)
fig = plot(X,Y)
savefig(fig, "A_low_cdf")
@info "mean(d) = $(Distributions.mean(d))"
@info "sd(d) = $(sqrt(Distributions.var(d)))"

sd = 0.2
a,b = AcsCertificates.Certificates.beta_parameters(1-mean, sd)
d = Distributions.Beta(a, b)
X = collect(range(0.0, 1.0, length=500))
Y = map(x -> Distributions.pdf(d, x), X)
fig = plot(X,Y)
savefig(fig, "A_high_pdf")
Y = map(x -> Distributions.cdf(d, x), X)
fig = plot(X,Y)
savefig(fig, "A_high_cdf")
@info "mean(d) = $(Distributions.mean(d))"
@info "var(d) = $(Distributions.var(d))"

mean = 0.9
sd = 0.05
a,b = AcsCertificates.Certificates.beta_parameters(1-mean, sd)
d = Distributions.Beta(a, b)
X = collect(range(0.0, 1.0, length=500))
Y = map(x -> Distributions.pdf(d, x), X)
fig = plot(X,Y)
savefig(fig, "B_low_pdf")
Y = map(x -> Distributions.cdf(d, x), X)
fig = plot(X,Y)
savefig(fig, "B_low_cdf")
@info "mean(d) = $(Distributions.mean(d))"
@info "var(d) = $(Distributions.var(d))"

sd = 0.1
a,b = AcsCertificates.Certificates.beta_parameters(1-mean, sd)
d = Distributions.Beta(a, b)
X = collect(range(0.0, 1.0, length=500))
Y = map(x -> Distributions.pdf(d, x), X)
fig = plot(X,Y)
savefig(fig, "B_high_pdf")
Y = map(x -> Distributions.cdf(d, x), X)
fig = plot(X,Y)
savefig(fig, "A_low_cdf")
@info "mean(d) = $(Distributions.mean(d))"
@info "var(d) = $(Distributions.var(d))"

mean = 0.7
sd = 0.05
a,b = AcsCertificates.Certificates.beta_parameters(1-mean, sd)
d = Distributions.Beta(a, b)
X = collect(range(0.0, 1.0, length=500))
Y = map(x -> Distributions.pdf(d, x), X)
fig = plot(X,Y)
savefig(fig, "C_low_pdf")
Y = map(x -> Distributions.cdf(d, x), X)
fig = plot(X,Y)
savefig(fig, "C_low_cdf")
@info "mean(d) = $(Distributions.mean(d))"
@info "var(d) = $(Distributions.var(d))"

sd = 0.3
a,b = AcsCertificates.Certificates.beta_parameters(1-mean, sd)
d = Distributions.Beta(a, b)
X = collect(range(0.0, 1.0, length=500))
Y = map(x -> Distributions.pdf(d, x), X)
fig = plot(X,Y)
savefig(fig, "C_high_pdf")
Y = map(x -> Distributions.cdf(d, x), X)
fig = plot(X,Y)
savefig(fig, "C_high_cdf")
@info "mean(d) = $(Distributions.mean(d))"
@info "var(d) = $(Distributions.var(d))"

AcsCertificates.Plots.acquisition("motivation", ["inverse", "improvement", "redistriction", "proportional"])
AcsCertificates.Plots.acquisition("A", ["proportional", "certification_A_low", "certification_A_high"])
AcsCertificates.Plots.acquisition("B", ["proportional", "proportional_estimate_B","certification_B_low", "certification_B_high"])
AcsCertificates.Plots.acquisition("C", ["proportional", "proportional_estimate_C","certification_C_low", "certification_C_high"])



