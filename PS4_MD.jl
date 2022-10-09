### PS4 ###
using Parameters, Plots, Statistics #import the libraries we want

#creating the capital grid which has more density at lower values
i_max = 75
i_min = 0
nk = 1000
inc = (i_max - i_min)/(nk - 1).^2  #increment
aux = collect(1:1:nk)
assets = zeros(length(aux))
for i = 1:length(aux)
    assets[i] = i_min + inc*(aux[i] - 1).^2
end
assets

@with_kw struct Primitives
    M::Int64 = 30  #the number of periods it takes to get to the new steady state 
    N::Int64 = 66 #live span of agents which is also the number of generations in our model
    n::Float64 = 0.011 #population growth rate 
    R::Int64 = 46 #Retirement age
    tw::Int64 = 45 #length of your working life
    tR::Int64 = N - (R+1) #length of your retirement
    θ::Float64 = 0.11 #labor income tax, when there's no social security θ = 0 and with social security θ = 0.11
    γ::Float64 = 0.42 #weight on consumption
    σ::Float64 = 2 #coefficient of relative risk aversion
    η::Array{Float64, 1} = [0.59923239, 
    0.63885106, 
    0.67846973, 
    0.71808840, 
    0.75699959,
    0.79591079, 
    0.83482198, 
    0.87373318,
    0.91264437, 
    0.95155556, 
    0.99046676, 
    0.99872065, 
    1.0069745, 
    1.0152284, 
    1.0234823, 
    1.0317362, 
    1.0399901, 
    1.0482440, 
    1.0564979, 
    1.0647518, 
    1.0730057, 
    1.0787834, 
    1.0845611, 
    1.0903388, 
    1.0961165, 
    1.1018943, 
    1.1076720, 
    1.1134497, 
    1.1192274, 
    1.1250052, 
    1.1307829, 
    1.1233544, 
    1.1159259, 
    1.1084974, 
    1.1010689, 
    1.0936404, 
    1.0862119, 
    1.0787834, 
    1.0713549, 
    1.0639264,
    1.0519200,
    1.0430000,
    1.0363000,
    1.0200000,
    1.0110000]
    z::Array{Float64, 1} = [3.0, 0.5] #productivity levels which can either be high z = 3 or low z = 0.5, with no risk z_h = z_l = 0.5
    markov::Array{Float64, 2} = [0.9261 0.0739; 0.0189 0.9811]  #markov matrix for productivity
    α::Float64 = 0.36  #capital share of output
    δ::Float64 = 0.06 #depreciation rate
    β::Float64 = 0.97 # discount factor
    a_grid::Array{Float64, 1} = assets #asset grid
    na::Int64 = length(a_grid) #length of the asset grid
    nz::Int64 = length(z) #length of productivity vector
    time::Array{Float64, 1} = collect(1:1:30)  #time vector that helps us keep track of the transition path
end


mutable struct Results
    val_func_::Array{Float64, 3} #value function, we need three dimensions because we need a value function for each age and productivity state
    pol_func::Array{Float64, 3} #policy function for assets
    labor::Array{Float64, 3}  #optimal labor supply
    
    #prices, capital, labor
    r::Float64 #interest rate
    w::Float64 #wage rate
    b::Float64  #social security benefit
    K_0::Float64 #aggregate capital intial guess
    L_0::Float64 #aggregate labor initial guess
end

function Initialize()
    prim = Primitives() #initialize primtiives
    val_func = zeros(prim.na, prim.nz, prim.N) #initial value function guess
    pol_func = zeros(prim.na, prim.nz, prim.N) #initial policy function for asset guess
    labor = zeros(prim.na, prim.nz, prim.tw) #inital guess of the optimal labor supply

    #### With social security ####
    K_0 = 3.364317  #aggregate capital with SS
    L_0 = 0.3432  #aggregate labor without SS
    w = 1.4557    #wage rate with SS
    r = 0.0235     #rental rate with SS
    b = 0.2253   #SS benefits

    ### Without social security  ####
    # K_0 = 4.406   #aggregate capital without SS
    # L_0 = 0.441   #aggregate labor without SS 
    # w = 1.4656
    # r = 0.08252
    # b = 0

    res = Results(val_func, pol_func, labor, r, w, b, K_0, L_0) #initialize results struct
    prim, res #return deliverables
end

#Initializing the population distribution 

mutable struct mu_results
    mu::Array{Float64, 1}
    mu_dist::Array{Float64, 3}
    mu1::Array{Float64, 3}
end

function Initialize_3()
    @unpack N, n, na, nz = prim
    mu_dist = zeros(prim.na, prim.nz, prim.N) #this will hold our unweighted distribution
    mu1 = zeros(prim.na, prim.nz, prim.N)  #this will hol our final, properly weight distribution
    mu = ones(prim.N)
    for i = 2:N
        mu[i] = mu[i-1]/(1 + n) #finding the relative sizes of each cohort (accounting for population growth)
    end
    mu = mu/sum(mu)  #normalizing mu so that it sums to 1
    mass = mu_results(mu, mu_dist, mu1)
end


# function Prices(prim::Primitives, res::Results, mass::mu_results)
#     @unpack α, δ, θ, R, N = prim
#     @unpack mu = mass
#     @unpack K_0, L_0 = res
#     w = (1-α)*(L_0)^(-α)*((K_0)^α)  #wage rate
#     r_k = α*(K_0)^(α - 1)*(L_0)^(1-α)  #the firm's interest rate
#     #Using the government budget constraint, we can solve for b
#     b = (θ*w*(L_0))/(sum(mu[R:N]))
#     r = r_k - δ  #the household's interest rate which is the firm's interest rate adjusted for depreciation
#     res.w = w
#     res.r = r
#     res.b = b
# end


#Solving the dynamic programming problem of retirees and workers

function Bellman(prim::Primitives, res::Results)
    @unpack N, R, tw, θ, γ, σ, η, z, markov, β, a_grid, na, nz = prim
    @unpack val_func, r, w, b = res
    
    #Solving the problem of retirees
    for a_index = 1:na, z_index = 1:nz #looping over assets and states
        c_N = (1 + r)*a_grid[a_index] + b    #last period consumption
        val_N = (c_N^((1-σ)*γ))/(1-σ)          #last period utility
        res.val_func[a_index, z_index, 66] = val_N #storing last period utility in the value function
        res.pol_func[a_index, z_index, 66] = 0.0 #storing the last period policy function
    end
    for j = (N-1):-1:R
        for a_index = 1:na, z_index = 1:nz  #looping over assets today
            a = a_grid[a_index] #setting the value of a
            candidate_max = -Inf  #initial guess of candidate max
            budget = (1 + r)*a + b #calculate the budget
            for ap_index = 1:na #looping over assets tomorrow
                c = budget - a_grid[ap_index] #consumption given a' selection
                if c>0  #check for positivity
                    val = (c^((1-σ)*γ))/(1-σ) + β * res.val_func[ap_index, z_index, j+1] #calculate the value function while looking at next period's value function
                    if val > candidate_max
                        candidate_max = val  #if our value function association with a' is greater than the current candidate max, update candidate max 
                        #to be equal to val. Then repeat the process for the next value of a' and update candidate max if this value of a' yields a higher
                        #value function. If the next value of a' does not yield a higher val, then candidate max will not update and we will just be left with
                        #the candidate max from before.
                        res.pol_func[a_index, z_index, j] = a_grid[ap_index] #update the policy function
                        res.val_func[a_index, z_index, j] = val  #updating the value function
                    end
                end
            end
        end
    end

    #Solving the problem of the working age people
    for i = tw:-1:1
        for a_index = 1:na  #looping over assets today
            a = a_grid[a_index] #setting the value of a
            for z_index = 1:nz #looping for productivity states
                candidate_max = -Inf  #initial guess of candidate max
                for ap_index = 1:na #looping over assets tomorrow
                    l = (γ*(1-θ)*(z[z_index] * η[i])*w - (1-γ)*((1+r)*a_grid[a_index] - a_grid[ap_index]))/((1-θ)*w*(z[z_index] * η[i]))  #for a given combination (z, a, a'), this is the 
                    #optimal labor supply
                    if l > 1  #this if, else loop ensures that labor supply is bounded between 0 and 1
                        l = 1
                    elseif l < 0
                        l = 0
                    end
                    budget = w*(1-θ)*(z[z_index] * η[i])*l + (1 + r)*a  #calculate the budget
                    c = budget - a_grid[ap_index] #consumption given a' selection
                        if c>0  #check for positivity
                            val = (((c^γ)*((1-l)^(1-γ)))^(1-σ))/(1-σ) + β * sum(res.val_func[ap_index, :, i+1].* markov[z_index, :]) #calculate the value 
                            #function while looking at next period's value function
                                if val > candidate_max
                                    candidate_max = val  #update candidate max
                                    res.pol_func[a_index, z_index, i] = a_grid[ap_index] #update the policy function
                                    res.val_func[a_index, z_index, i] = val  #updating the value function
                                    res.labor[a_index, z_index, i] = l  #updating the optimal labor supply vector 
                                end
                        end
                end
            end
        end
    end
end

function Distribution(prim::Primitives, res::Results, mass::mu_results)
    @unpack N, R, tw, z, a_grid, na, nz, markov = prim
    @unpack pol_func = res
    @unpack mu, mu_dist, mu1 = mass

    mu_dist[1, 1, 1] = 0.2037  #initial mass of high productivity people
    mu_dist[1, 2, 1] = 0.7963   #initial mass of low productivity people
    for j = 2:prim.N  #loopig over the ages
        for ap = 1:prim.na  #looping over assets tomorrow
            for z = 1:2  #looping over z from today
                d = findall(x->x == a_grid[ap], res.pol_func[:, z, j-1])
                    for i = 1:length(d) #loop over the indices you find for a specific z
                        for zp = 1:2  #loop over zp 
                            mu_dist[ap, zp, j] = mu_dist[ap, zp, j] + mu_dist[d[i], z, j-1] .* markov[z, zp]
                        end
                    end
            end
        end
    end

    #The final step is weighting each generation according to the relative sizes calculated by mu.
    for i = 1:prim.N
        mu1[:, :, i] = mu_dist[:, :, i] .* mu[i]
    end
    mass.mu_dist = mu_dist  #updating the unweighted distribution
    mass.mu1 = mu1  #updating the weighted distribution 
end

function Solve_model(prim::Primitives, res::Results, mass::mu_results)
    Prices(prim, res, mass)   #Define prices by solving the firm's problem and GBC
    println("Capital is", res.K_0)
    println("Labor is", res.L_0)
    Bellman(prim, res) #solve the Bellman!
    Distribution(prim, res, mass)   #solve for the stationary distribution
end

prim, res = Initialize()
mass = Initialize_3()
Solve_model(prim, res, mass)

#Calculating total welfare

welfare = zeros(prim.na, prim.nz, prim.N)  #an array holding inidividual welfare 
for i = 1:prim.N   #looping over ages 
    for a = 1:prim.na   #looping over asset holdings
        for z = 1:prim.nz  #looping over states
            welfare[a, z, i] = res.val_func[a, z, i] .* mass.mu1[a, z, i]   #I multiply each value in the value function by the mass of people actually
            #at that level of asset holding and productivity state in the population distribution
        end
    end
end

total_welfare = sum(welfare)   #to get total welfare, I just sum all of the individual welfares
#total welfare is W = -35.763

#Now we calculate the coefficient of welath variation cv = σ/μ where σ is the variance of the population wealth and
# μ is the mean of the wealth. 

wealth_dist = zeros(prim.na, prim.nz, prim.N)
#wealth for the young who earn labor income,   budget = w*(1-θ)*(z[z_index] * η[i])*l + (1 + r)*a
for j = 1:prim.tw  #age
    for a = 1:prim.na  #assets
        for i = 1:prim.nz  #pick a productivity state
            wealth_dist[a, i, j]= mass.mu1[a,i,j]*(res.labor[a,i,j] * res.w * (1 - prim.θ) * prim.z[i]*prim.η[j] + (1 + res.r) * prim.a_grid[a])
        end
    end
end
#wealth of the retired people,  budget = (1 + r)*a + b
for j = prim.R:prim.N  #looping over the retired ages
    for a = 1:prim.na  #assets
        for i = 1:prim.nz  #state
            wealth_dist[a, i, j] = mass.mu1[a,i,j]*((1+res.r)*prim.a_grid[a] + res.b)
        end
    end
end

μ = mean(wealth_dist)
σ = var(wealth_dist)
cv = (σ/μ)*100
#cv = 0.2553