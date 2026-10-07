using Random
using LinearAlgebra



function P(x, y, v)
	"""
	Parallel transport operator on the sphere
	"""
	if x != -y
		return v .- (x .+ y) .* (dot(y, v)/(1 + dot(x,y)))
	else
		#special case for when x and y are antipodal
		local pₛ = zeros(length(x))
		pₛ[end] = -1.
		return v .- 2. .* (pₛ .- dot(pₛ,x) .* x) .* (dot(pₛ, v)/(1 - dot(x, pₛ)^2))
	end
end

function M(x, y, v)
	"""
	Parallel transport composed with reflection operator on the sphere
	"""
	if x == y
		return v
	elseif x != -y
		return v .- (x .+ y) .* (dot(y, v)/(1 + dot(x,y))) .+ (2*dot(y .- (x⋅y).*x, v)/(1 - dot(x,y)^2)) .* (x .- (x⋅y).*y)
	else
		#special case for when x and y are antipodal
		local pₛ = zeros(length(x))
		pₛ[end] = -1.
		return v .- 2. .* (pₛ .- dot(pₛ,x) .* x) .* (dot(pₛ, v)/(1 - dot(x, pₛ)^2)) .+ (2*dot(pₛ .- (x⋅pₛ).*x, v)/(1 - dot(x,pₛ)^2)) .* (pₛ .- (y⋅pₛ).*y)
	end

end

function get_e_x_perp_y(x, y)
	"""
	Compute the unit vector in the tangent space of x pointing to y
	"""

	if dot(x,y) >= 1.
		return x .- y
	elseif dot(x,y) > -1
		return (y .- dot(x,y) .* x)./sqrt(1 - dot(x,y)^2)
	else
		#special case for when x and y are antipodal in which case all unit vectors in the tangent 
		#space of x point to y so we choose the one pointing to the south pole
		local pₛ = zeros(length(x))
		pₛ[end] = -1.
		return (pₛ .- dot(x,pₛ) .* x)./sqrt(1 - dot(x,pₛ)^2)
	end
end

function sdist(x,y)
	"""
	Compute geodesic distance between two points on the sphere
	"""
	return acos(max(min(dot(x,y), 1.), -1.))
end

function stereo_projection(x; R)
	return  R .* @view(x[1:end-1])./ (1 - x[end])
end

function spherical_log_accept_ratio(x, x_prop, log_density, R, d)
	"""
	Logarithm of the acceptance ratio on the sphere
	"""
	return log_density(stereo_projection(x_prop; R = R)) - log_density(stereo_projection(x; R = R)) + d*(log(1 - x[end]) - log(1-x_prop[end]))
end



"""
Defining functions for one step of the random walk and coupled random walks on the sphere
"""

function stereo_rw_step!(x, h, G)
	"""

	"""
	x .+= h .* (G .- dot(G, x).* x)
	normalize!(x)
	return x
end

function parallel_coupling_step!(x, y, G, G̃, h)
	"""

	"""
	G .-= dot(G, x) .* x #projecting noise onto tangent space of x
	G̃ .= P(x, y, G) #parallel transporting noise to tangent space of y
	x .+= h .* G
	y .+= h .* G̃
	normalize!(x)
	normalize!(y)

	return x, y
end

function reflection_coupling_step!(x, y, G, G̃, h)
	"""

	"""
	G .-= dot(G, x) .* x #projecting noise onto tangent space of x
	G̃ .= M(x, y, G) #parallel transporting noise to tangent space of y and reflecting it
	x .+= h .* G
	y .+= h .* G̃
	normalize!(x)
	normalize!(y)

	return x, y
end

@inline function log_meeting_proba(x, y, X_new, h, d)
	"""
	Compute logarithm of the ratio of Q(y, X_new) and Q(x, X_new) where Q is the transition density
	of the stereo random walk with step size h and dimension d.
	"""
	return dot(y, X_new) > 0 ? (d+1)*log(dot(x, X_new)/dot(y, X_new)) + (1/dot(x, X_new)^2 - 1/dot(y, X_new)^2)/(2h^2) : -Inf
end

function max_reflection_coupling_step!(x, y, X_new, G, G̃, h, d, RNG)
	"""

	"""
	G .-= dot(G, x) .* x #projecting noise onto tangent space of x
	G̃ .= M(x, y, G) #parallel transporting noise to tangent space of y and reflecting it

	X_new .= x
	X_new .+= h .* G
	normalize!(X_new)

	if (dot(y, X_new) >= 0.) && (rand(RNG) >= -log_meeting_proba(x, y, X_new, h, d))
		#meeting event
		y .= X_new
	else
		#reflection event
		y .+= h .* G̃
		normalize!(y)
	end
	x .= X_new

	return x, y
end


"""
Metreopolised transition steps
"""

function stereo_MH_step!(x, x_prop, G, h, d, R, log_density, RNG)
	stereo_rw_step!(x_prop, h, G) #proposed move

	if randexp(RNG) >= -spherical_log_accept_ratio(x, x_prop, log_density, R, d)
		x .= x_prop #accept move
	else
		x_prop .= x #reject move
	end
	return x, x_prop
end


function sync_coupled_MH_step!(x, x_prop, y, y_prop, G, G̃, h, d, R, log_density, RNG)
	parallel_coupling_step!(x, y, G, G̃, h) #proposed move

	E = randexp(RNG)

	if E >= -spherical_log_accept_ratio(x, x_prop, log_density, R, d)
		x .= x_prop #accept move
	else
		x_prop .= x #reject move
	end
	if E >= -spherical_log_accept_ratio(y, y_prop, log_density, R, d)
		y .= y_prop #accept move
	else
		y_prop .= y #reject move
	end
	
	return x, x_prop, y, y_prop
end

function coupled_MH_step!(x, x_prop, y, y_prop, X_new, G, G̃, h, d, R, log_density, RNG)
	max_reflection_coupling_step!(x_prop, y_prop, X_new, G, G̃, h, d, RNG) #proposed move

	E = randexp(RNG)

	x_accepted = false
	y_accepted = false

	if E >= -spherical_log_accept_ratio(x, x_prop, log_density, R, d)
		x .= x_prop #accept move
		x_accepted = true
	else
		x_prop .= x #reject move
	end
	if E >= -spherical_log_accept_ratio(y, y_prop, log_density, R, d)
		y .= y_prop #accept move
		y_accepted = true
	else
		y_prop .= y #reject move
	end

	return x_accepted, y_accepted
end