##### Exercise 2 #######
# the goverment annouces that at time t = 21, they are going to eliminate social security
using Parameters, Plots, Statistics #import the libraries we want

#Solving the SS model and storing the distribution and value functions
include("PS4_SS_MD.jl")

prim, res = Initialize()
mass = Initialize_3()
Solve_model(prim, res, mass)

@unpack mu1, mu_dist = mass
@unpack val_func = res
distribution_SS = mass.mu1
value_func_SS = res.val_func

#Solving the model without S.S.
include("PS4_NO_SS_MD.jl")

prim, res_NSS = Initialize_NSS()
mass_NSS = Initialize_3_NSS()
Solve_model(prim, res_NSS, mass_NSS)

@unpack mu1_NSS = mass_NSS
@unpack val_func = res_NSS
distribution_NSS = mass_NSS.mu1_NSS
value_func_NSS = res_NSS.val_func

mutable struct transition 
    k_t::Array{Float64, 1}  #the path of aggregate capital
    l_t::Array{Float64, 1}  #the path of aggregate labor 
    r_t::Array{Float64, 1}  #path of the rental rate
    w_t::Array{Float64, 1}  #path of the wage rate 
    pol_func_t::Array{Float64, 4}  #savings decision rule induced by the transition path
    val_func_t::Array{Float64, 4}  #value functions along the transition path
    labor_t::Array{Float64, 4}   #labor decision rules along the transition path
    θ_t::Array{Float64, 1}  #theta along the transiton path
    mu_t::Array{Float64, 4}  #cross section distribution along the transition path
    M::Int64
    mu::Array{Float64, 1} #weighting distributions 
    b_t::Array{Float64, 1}  #social security benefits
end

function Initialize_t()
    prim = Primitives()  #initialize Primitives
    M = 60  #length of time horizon 
    k_t = collect(range(3.364317, length = M, stop = 4.483))  #initial guess of aggregate capital
    l_t = collect(range(0.3432, length = M, stop = 0.3634))  #initial guess of aggregate labor path
    r_t = zeros(M)  #initial guess of interest rate path
    w_t = zeros(M)  #guess of wage rate path
    pol_func_t = zeros(prim.na, prim.nz, prim.N, M)
    val_func_t = zeros(prim.na, prim.nz, prim.N, (M+1)) #we add an additional time period to the value function to hold the No SS equilibrium
    for i = 2:M    #for the no S.S. states, we need to initialize with -50
        val_func_t[:, :, :, i] .= -50
    end
    labor_t = zeros(prim.na, prim.nz, prim.N, M)
    θ_t = zeros(M)
    for i = 1:21
        θ_t[i] = 0.11  # the labor tax is only equal to 0.11 in the first 21st time periods
    end
    b_t = zeros(M)   #social security benefits over time
    mu_t = zeros(prim.na, prim.nz, prim.N, M)
    mu = ones(prim.N)
        for i = 2:prim.N
            mu[i] = mu[i-1]/(1 + prim.n) #finding the relative sizes of each cohort (accounting for population growth)
        end
    mu = mu/sum(mu)  #normalizing mu so that it sums to 1
    output = transition(k_t, l_t, r_t, w_t, pol_func_t, val_func_t, labor_t, θ_t, mu_t, M, mu, b_t)
    prim, output #return deliverables
end

function Prices_t(prim::Primitives, output::transition) 
    @unpack α, δ = prim
    @unpack k_t, l_t, r_t, w_t, M, θ_t, mu = output
    r_t = zeros(M) 
    for i = 1:length(r_t)  #calculatig the interest rate path using the firm's problem
        r_t[i] = (α*(k_t[i])^(α - 1))*((l_t[i])^(1-α)) - δ   #household interest rate
    end
    w_t = zeros(M)
    for i = 1:length(w_t) #calculating the wage rate path using the firm's problem
        w_t[i] = (1-α)*((l_t[i])^(-α))*((k_t[i])^α)  #wage rate
    end
    b_t = zeros(M)
    for i = 1:length(b_t) 
        b_t[i] = (θ_t[i] * w_t[i] * l_t[i])/(sum(mu[prim.R:prim.N]))
    end
    output.r_t = r_t  #update the interest rate path
    output.w_t = w_t  #update the wage rate path
    output.b_t = b_t  #update social security 
end


function Dynamic_Programming(prim::Primitives, output::transition)
    @unpack N, R, tw, γ, σ, η, z, markov, β, a_grid, na, nz, α, δ = prim
    @unpack val_func_t, r_t, w_t, θ_t, pol_func_t, labor_t, M, b_t = output

    #First, we store the value functions and distribution for the last period where we're in the no ss steady state
    val_func_t[:, :, :, M+1] = value_func_NSS

    for time = M:-1:1   #looping over each time period that we don't currently know the values for in reverse order
        θ = θ_t[time]  #the theta in the household's problem will correspond with the theta in that time period
        b = b_t[time]  #social security for this time period
        #Now that we have prices, we solve the household's problems using the Bellman for No SS because there's no social security during these time periods
        for a_index = 1:na, z_index = 1:nz #looping over assets and states
            r = r_t[time]  #setting the value of r
            c_N = (1 + r)*a_grid[a_index] + b  #last period consumption
            val_N = (c_N^((1-σ)*γ))/(1-σ)    #last period utility
            if val_N > -50
                output.val_func_t[a_index, z_index, 66, time] = val_N #storing last period utility in the value function
                output.pol_func_t[a_index, z_index, 66, time] = 0.0 #storing the last period policy function
            elseif val_N < -50
                output.val_func_t[a_index, z_index, 66, time] = 0.0 #storing last period utility in the value function
                output.pol_func_t[a_index, z_index, 66, time] = 0.0 #storing the last period policy function
            end
        end
        for j = (N-1):-1:R
            for a_index = 1:na, z_index = 1:nz  #looping over assets today
                a = a_grid[a_index] #setting the value of a
                r = r_t[time]  #setting the value of r
                candidate_max = -Inf  #initial guess of candidate max
                budget = (1 + r)*a + b #calculate the budget
                for ap_index = 1:na #looping over assets tomorrow
                    c = budget - a_grid[ap_index] #consumption given a' selection
                    if c>0  #check for positivity
                        val = (c^((1-σ)*γ))/(1-σ) + β * output.val_func_t[ap_index, z_index, j+1, time + 1] #calculate the value function while looking at next period's value function
                        if val > candidate_max
                            candidate_max = val  
                            output.pol_func_t[a_index, z_index, j, time] = a_grid[ap_index] #update the policy function
                            output.val_func_t[a_index, z_index, j, time] = val  #updating the value function
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
                    w = w_t[time]  #fixing the wage rate to be the wage rate for this time period
                    r = r_t[time]  #fixing the interest rate 
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
                                val = (((c^γ)*((1-l)^(1-γ)))^(1-σ))/(1-σ) + β * sum(output.val_func_t[ap_index, :, i+1, time + 1].* markov[z_index, :]) #calculate the value 
                                #function while looking at next period's value function
                                    if val > candidate_max
                                        candidate_max = val  #update candidate max
                                        output.pol_func_t[a_index, z_index, i, time] = a_grid[ap_index] #update the policy function
                                        output.val_func_t[a_index, z_index, i, time] = val  #updating the value function
                                        output.labor_t[a_index, z_index, i, time] = l  #updating the optimal labor supply vector 
                                    end
                            end
                    end
                end
            end
        end  
    end #end of the HH's problems
end #End of DP function 


#Now we can calculate the distribution for time t using the policy functions derived above. 
function Distribution_t(prim::Primitives, output::transition)
    @unpack na, nz, N, a_grid, markov, n = prim
    @unpack pol_func_t, M, mu = output
    mu_t = zeros(prim.na, prim.nz, prim.N, M)
    #First, we save the initial distribution as being equal to the one from the steady state with social security
    mu_t[:, :, :, 1] = distribution_SS
    mu_dist = zeros(prim.na, prim.nz, prim.N, M)
    mu_dist[:, :, :, 1] = mass.mu_dist
    for time = 2:M  #now we can go in forward order  
        mu_dist[1, 1, 1, time] = 0.2037  #initial mass of high productivity people which will occur for each time period since we have a new generation being born
        mu_dist[1, 2, 1, time] = 0.7963   #initial mass of low productivity people
        for j = 2:prim.N  #loopig over the ages
            for ap = 1:prim.na  #looping over assets tomorrow
                for z = 1:2  #looping over z from today
                    d = findall(x->x == a_grid[ap], output.pol_func_t[:, z, j-1, time-1]) #want to look at policy fucntion for the previous time period
                        for i = 1:length(d) #loop over the indices you find for a specific z
                            for zp = 1:2  #loop over zp 
                                mu_dist[ap, zp, j, time] = mu_dist[ap, zp, j, time] + mu_dist[d[i], z, j-1, time-1] .* markov[z, zp]
                            end
                        end
                end
            end
        end

        #The final step is weighting each generation according to the relative sizes calculated by mu.

        for i = 1:prim.N
            mu_t[:, :, i, time] = mu_dist[:, :, i, time] .* mu[i]
        end
    end  #ending the time loop
    output.mu_t = mu_t #updating the weighted distribution time path
end



#Now we calculate aggregate capital and labor for each time period
function Aggregate(prim::Primitives, output::transition)
    @unpack na, N, a_grid, z, η, tw, nz = prim
    @unpack mu_t, labor_t, k_t, l_t, M = output
    k_t_new = zeros(M)
    k_t_new[1] = 3.364317
    l_t_new = zeros(M)
    l_t_new[1] = 0.3432
    agg_capital = zeros(prim.na, prim.N, M)
    for t = 2:M #we only need to update aggregate capital and labor after the first period
        for i = 1:prim.N  #pick a specific age
            for j = 1:prim.na  #pick a specific asset level holding
                agg_capital[j, i, t] = sum(mu_t[j, :, i, t]) .* a_grid[j]
            end
        end
    k_t_new[t] = sum(agg_capital[:, :, t])
    end

    #Calculating aggregate labor 
    #begin by calculating e(z, η_j) as vector
    e = zeros(prim.tw, prim.nz)
    for i = 1:prim.tw
        for j = 1:prim.nz
            e[i, j] = prim.z[j] * prim.η[i]
        end
    end

    agg_l = zeros(prim.na, prim.tw, M)  
    for t = 2:M 
        for i = 1:prim.tw  #pick a generation
            for a = 1:prim.na  #pick an asset holding
                for j = 1:prim.nz  #pick a productivity state
                    agg_l[a, i, t]= sum(mu_t[a,:,i,t].*e[i,:].*labor_t[a,:,i,t])
                end
            end
        end
        l_t_new[t] = sum(agg_l[:, :, t])
    end #ending time loop
    return agg = vcat(k_t_new', l_t_new')  #the function will return an array where the first row is the path of agregate capital and the 
    #second row is the path of aggregate labor 
end #ending aggregate function



prim, output = Initialize_t()
# Prices_t(prim, output)
# output.θ_t
# output.b_t

function Solve_model_t(prim::Primitives, output::transition)
    Prices_t(prim, output) #first we solve for the paths of r and w
    Dynamic_Programming(prim, output)  #then we solve the HH problems to get the policy functions
    Distribution_t(prim, output)  #then we solve for the path of our population distribution using the policy function for a'
    Aggregate(prim, output)  #finally we calculate the path of aggregate capital and labor
end
Solve_model_t(prim, output)

mutable struct results_path
    K_final::Array{Float64, 1}
    L_final::Array{Float64, 1}
    val_final::Array{Float64, 4}
end
function Results_t()
    K_final = zeros(output.M)
    L_final = zeros(output.M)
    val_final = zeros(prim.na, prim.nz, prim.N, output.M + 1)
    res_t = results_path(K_final, L_final, val_final)
    res_t
end
res_t = Results_t()

function Iterate_t(prim::Primitives, output::transition; tol::Float64 = 0.001)
    @unpack k_t, l_t, M = output  #unpack initial guess of k and l path. Then we'll solve the model once before iterating
    prim, output = Initialize_t()
    Solve_model_t(prim, output)
    K_1 = Aggregate(prim, output)[1, :]   #the new aggregate capital path which comes from market clearing
    L_1 = Aggregate(prim, output)[2, :]   #the new aggregate labor path 

    #calculate the errors for capital and labor separately
    err_k = Aggregate(prim, output)[1,:] - output.k_t
    err_l = Aggregate(prim, output)[2,:] - output.l_t
    n = 0  #counter

    while n < 25   #I'm limiting the while loop to only 25 iterations to avoid it looping forever like last time 
        println("on iteration", n)
        if maximum(err_k) + maximum(err_l) > tol
            output.k_t = 0.5K_1 + (1 - 0.5)output.k_t
            output.l_t = 0.5L_1 + (1 - 0.5)output.l_t

            Solve_model_t(prim, output)  #solve the model again
            K_1 = Aggregate(prim, output)[1, :]   
            L_1 = Aggregate(prim, output)[2, :] 

            #update the errors
            err_k = Aggregate(prim, output)[1, :] - output.k_t
            err_l = Aggregate(prim, output)[2, :] - output.l_t

            println("Capital path now", K_1)
            println("Labor path now", L_1)
            res_t.K_final = K_1
            res_t.L_final = L_1
            res_t.val_final = output.val_func_t
       
        elseif maximum(err_k) + maximum(err_l) < tol
            println("it converged!")
        end
        n+=1  #update the counter 
    end  #end the while loop
end #end the iterate function

Iterate_t(prim, output)

@unpack K_final, L_final, val_final = res_t
error_k = K_final[60] - 4.482
error_l = L_final[60] - 0.3634
output.k_t = K_final   #updating the capital path
output.l_t = L_final   #updating the labor path
Prices_t(prim, output)  #updating prices
Dynamic_Programming(prim, output)  #then we solve the HH problems to get the policy functions
Distribution_t(prim, output)

####  Repeating Exercise 1  ####
# Plot the transition paths of the interest rate, wage rate, capital and effective labor
@unpack k_t, l_t, r_t, w_t, val_func_t, b_t = output
t_grid = collect(1:1:output.M)  #creating a time grid to use as my x-axis on my graphs
Plots.plot(t_grid, k_t, title="Capital Transition Path")
Plots.savefig("Capital_transition_path_2.png")
#There's a very small kink after the change in policy at t=21 where the graph shifts from being linear
#to being concave.
Plots.plot(t_grid, l_t, title="Labor Transition Path")
Plots.savefig("Labor_transition_path_2.png")
#There's a large initial increase in labor following the removal of social security
#which likely reflects agents trying to compensate for the lack of future income by working
#more now so that they will have savings for retirement. Overtime, aggregate labor starts to 
#decline. There's also a jump in labor initially after the announcement is made that the
#social security system will be eliminated. 
Plots.plot(t_grid, r_t, title="Interest rate Transition Path")
Plots.savefig("Interest_rate_transition_path_2.png")
#The interest rate has a sharp increase after social security is eliminated, then
#steadily decreases overtime. This makes sense because the removal of social security
#results in more aggregate capital over time which drives down the interest rate.
Plots.plot(t_grid, w_t, title="Wage rate Transition Path")
Plots.savefig("Wage_rate_transition_path_2.png")
#The wage rate experiences a steep, small decline immediately after the removal corresponding
#to the steep initial increase in labor. However, overtime the wage rate rises.


# Question 3: what fraction of the overall population would support the reform? Compute and plot
#the measure of comsumption equivalent variation for each age EVj

#Storing the value functions for the generations in the initial steady state
val_transition_0 = val_final[:, :, :, 1]
#In the model we solved for PS3, the value function in the steady state with SS is 
value_func_SS

# calculating EV(a, z, j) for each generation j
EV = zeros(prim.na, prim.nz, prim.N)
for j = 1:prim.N  #loop over generations
    for a = 1:prim.na   #loop over assets
        for z = 1:prim.nz   #loop over states
            EV[a, z, j] = (val_transition_0[a, z, j]/value_func_SS[a, z, j])^(1/(prim.γ * (1 - prim.σ)))
        end
    end
end
EV

# Consumption equivalent variation 
EV_1 = zeros(prim.na, prim.nz, prim.N)
for j = 1:prim.N 
    for a = 1:prim.na
        for z = 1:prim.nz
            EV_1[a, z, j] = EV[a, z, j] * output.mu_t[a, z, j, 1]
        end
    end
end
EV_1


EV_j = zeros(prim.N)   #computing EV for each generation by suming over a and z 
EV_j[1] = sum(EV_1[:, :, 1])
for i = 2:prim.N 
    EV_j[i] = sum(EV_1[:, :, i])
end
EV_j   #this is the vector of consumption equivalent variation for each generation

j_grid = collect(1:1:prim.N)  #x-axis for EV graph 
Plots.plot(j_grid, EV_j, title="Consumption equivalent variation")
Plots.savefig("consumption_equivalent_variation_2.png")
#Equivalent variation: now that prices have changed, how much income would we have to give you
#to be able to afford your old level of utility?
# EV decreases with each generation. This is because the young are much better off in the 
# no social security world and value that more than older people who are facing retirement much
#sooner (or are already retired) and have very little time to save for their old age. Their value
#of this reform is much lower. We'd expect to see the young voting for the reform while the old 
#would vote against it. The retirees would definitely not support it. 